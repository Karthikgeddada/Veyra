"""LangGraph workflow for DevTeam AI (Forge).

Extends the original iterative pipeline with an interactive tool loop:

    Requirement Analyzer -> Architect -> Developer -> Validate ->
    Tester -> Forge runs tests (run_command, with human approval) ->
    (Debugger -> run tests)?  -> Reviewer -> Documentation -> Packager

The test step is now a real command execution through the controlled
``ProjectTools.run_command`` tool. When the command policy requires
approval and the run is interactive, the node pauses the graph with
LangGraph ``interrupt()`` and resumes once the user decides (allow once /
allow session / deny). A MemorySaver checkpointer makes pause/resume
possible. The debug loop is still bounded by ``MAX_AGENT_ITERATIONS``.

State lists that accumulate across nodes (logs, agent_messages,
chat_messages, tool_calls, tool_results, approved_commands) use
``Annotated[list, add]`` reducers so they survive checkpointing; scalar /
replace fields keep default semantics. Existing fields are unchanged.
"""

from operator import add
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from agents.architect_agent import architect_agent
from agents.coder_agent import coder_agent
from agents.debugger_agent import debugger_agent
from agents.documentation_agent import documentation_agent
from agents.forge_agent import forge_summarize
from agents.packager_agent import packager_agent
from agents.requirement_agent import requirement_agent
from agents.reviewer_agent import reviewer_agent
from agents.tester_agent import tester_agent
from config.llm_config import (
    MAX_AGENT_ITERATIONS,
    MAX_COMMAND_EXECUTIONS,
    MAX_DEBUG_ITERATIONS,
    MAX_LLM_CALLS,
    get_model_for_task,
    list_task_models,
)
from services.command_policy import (
    CommandRisk,
    classify_command,
    command_signature,
    needs_approval,
)
from services.tools import ProjectTools, validate_files

# Human-in-the-loop primitives (interrupt/Command + checkpointer). These are
# only present in langgraph >= 0.2.31; the workflow degrades gracefully when
# they are unavailable (non-interactive, auto-approve safe commands).
try:  # pragma: no cover - import availability depends on installed version
    from langgraph.types import Command, interrupt
    from langgraph.checkpoint.memory import MemorySaver

    HITL_AVAILABLE = True
except Exception:  # noqa: BLE001
    Command = None  # type: ignore[assignment]
    interrupt = None  # type: ignore[assignment]
    MemorySaver = None  # type: ignore[assignment]
    HITL_AVAILABLE = False


class DevState(TypedDict):
    run_id: str
    prompt: str
    requirements: str
    architecture: str
    code_files: List[Dict[str, str]]  # [{"path": "...", "content": "..."}]
    review_comments: str
    tests_files: List[Dict[str, str]]
    docs_files: List[Dict[str, str]]
    status: str
    logs: Annotated[List[str], add]
    # --- enriched state (Milestone 1) ---
    current_agent: str
    agent_messages: Annotated[List[Dict[str, str]], add]
    test_results: str
    errors: str
    iteration_count: int
    final_status: str
    project_spec: str
    files_modified: List[str]
    usage_info: Dict[str, Any]
    # --- Forge interactive state (Milestone 2) ---
    interactive: bool
    auto_approve_safe: bool
    chat_messages: Annotated[List[Dict[str, Any]], add]
    tool_calls: Annotated[List[Dict[str, Any]], add]
    tool_results: Annotated[List[Dict[str, Any]], add]
    pending_approval: Dict[str, Any]
    approval_status: str
    requested_command: str
    command_output: str
    command_error: str
    command_exit_code: int
    command_timeout: bool
    current_tool: str
    approved_commands: Annotated[List[str], add]
    # --- Forge bounded execution + project understanding (Milestone 3) ---
    llm_calls: int
    command_executions: int
    project_analysis: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _initial_usage() -> Dict[str, Any]:
    return {
        "agent_calls": 0,
        "llm_calls": 0,
        "command_executions": 0,
        "models": list_task_models(),
        "max_iterations": MAX_AGENT_ITERATIONS,
        "max_llm_calls": MAX_LLM_CALLS,
        "max_command_executions": MAX_COMMAND_EXECUTIONS,
    }


def _activity_entry(agent: str, message: str, task_type: str = "default") -> Dict[str, str]:
    return {"agent": agent, "message": message, "model": get_model_for_task(task_type)}


def _bump_usage(state: DevState) -> Dict[str, Any]:
    usage = dict(state.get("usage_info") or _initial_usage())
    usage["agent_calls"] = usage.get("agent_calls", 0) + 1
    usage["llm_calls"] = usage.get("llm_calls", 0) + 1
    return usage


def _forge_msg(text: str, tool_card: Any = None) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"role": "forge", "agent": "Forge", "content": text}
    if tool_card is not None:
        msg["tool_card"] = tool_card
    return msg


def _user_msg(text: str) -> Dict[str, str]:
    return {"role": "user", "content": text}


def _project_root(run_id: str) -> str:
    return f"generated_projects/{run_id}"


def make_config(thread_id: str) -> Dict[str, Any]:
    """LangGraph config with a thread_id (required by the checkpointer)."""
    return {"configurable": {"thread_id": thread_id}}


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
async def node_understand(state: DevState):
    """Understand the existing project structure (deterministic, no LLM).

    Runs :meth:`ProjectTools.analyze_project` to detect the tech stack,
    list files, and set up the project context. This node runs before
    the requirement analyzer so Forge can adapt to existing projects.
    """
    run_id = state.get("run_id", "default_run")
    tools = ProjectTools(_project_root(run_id))
    try:
        analysis = tools.analyze_project()
    except Exception as exc:  # noqa: BLE001
        analysis = {"error": str(exc), "stack": {}}
    stack = analysis.get("stack", {})
    lang = stack.get("language", "unknown")
    framework = stack.get("framework", "unknown")
    files_count = analysis.get("filtered_files", 0)
    log_msg = f"Project analyzed: {files_count} files, language={lang}, framework={framework}"
    forge_msg = (
        f"I've analyzed the project workspace. I found {files_count} file(s) "
        f"using {lang}" + (f" with {framework}" if framework != "unknown" else "") + ". "
        f"Let me now analyze your requirements."
    )
    return {
        "status": "understanding_project",
        "current_agent": "Project Analyzer",
        "project_analysis": analysis,
        "logs": [log_msg],
        "agent_messages": [_activity_entry("Project Analyzer", log_msg, "fast")],
        "usage_info": dict(state.get("usage_info") or _initial_usage()),
        "chat_messages": [_forge_msg(forge_msg)],
    }


async def node_requirements(state: DevState):
    new_logs = ["Analyzing requirements..."]
    reqs = await requirement_agent(state["prompt"])
    usage = _bump_usage(state)
    return {
        "requirements": reqs,
        "status": "analyzing_requirements",
        "project_spec": reqs,
        "current_agent": "Requirement Analyzer",
        "logs": new_logs,
        "agent_messages": [_activity_entry("Requirement Analyzer", "Requirements extracted", "planning")],
        "usage_info": usage,
        "chat_messages": [_user_msg(state["prompt"]), _forge_msg("Let me start by analyzing your requirements.")],
    }


async def node_architecture(state: DevState):
    arch = await architect_agent(state["prompt"], state["requirements"])
    usage = _bump_usage(state)
    return {
        "architecture": arch,
        "status": "designing_architecture",
        "current_agent": "Architect",
        "logs": ["Designing architecture..."],
        "agent_messages": [_activity_entry("Architect", "Architecture designed", "planning")],
        "usage_info": usage,
        "chat_messages": [_forge_msg("Designing the project architecture.")],
    }


async def node_coder(state: DevState):
    files = await coder_agent(state["prompt"], state["architecture"])
    usage = _bump_usage(state)
    return {
        "code_files": files,
        "files_modified": [f.get("path", "") for f in files],
        "status": "generating_code",
        "current_agent": "Developer",
        "logs": ["Generating code files..."],
        "agent_messages": [_activity_entry("Developer", f"Generated {len(files)} files", "coding")],
        "usage_info": usage,
        "chat_messages": [_forge_msg(f"I've generated {len(files)} file(s). Now I'll validate and run the tests.")],
    }


async def node_validate(state: DevState):
    """Deterministic structural validation (no LLM call)."""
    ok, message = validate_files(state.get("code_files", []))
    logs = ["Validating generated files..."]
    errors = ""
    chat: List[Dict[str, Any]] = []
    if not ok:
        logs.append(f"Validation issues: {message}")
        errors = message
        chat.append(_forge_msg(f"Structural validation found issues: {message}"))
    else:
        logs.append("Structural validation passed.")
        chat.append(_forge_msg("Structural validation passed."))
    return {
        "status": "validating",
        "current_agent": "Validator",
        "errors": errors,
        "logs": logs,
        "chat_messages": chat,
    }


async def node_tester(state: DevState):
    tests = await tester_agent(state["code_files"])
    usage = _bump_usage(state)
    return {
        "tests_files": tests,
        "status": "testing",
        "current_agent": "Tester",
        "logs": ["Generating tests..."],
        "agent_messages": [_activity_entry("Tester", f"Generated {len(tests)} test files", "coding")],
        "usage_info": usage,
        "chat_messages": [_forge_msg(f"I wrote {len(tests)} test file(s). Let me run them.")],
    }


async def node_run_tests(state: DevState):
    """Run the test suite via the controlled run_command tool.

    Pauses the graph with ``interrupt()`` for human approval when the command
    policy requires it (interactive mode only). Non-interactive runs
    auto-execute safe commands and skip approval-required ones.
    """
    run_id = state.get("run_id", "default_run")
    command = "python -m pytest -q"
    reason = "Run the generated test suite."
    working_directory = "."
    tool_name = "run_command"

    tools = ProjectTools(_project_root(run_id))
    # Materialize code + test files so the runner can find them on disk.
    tools.write_files(state.get("code_files", []) + state.get("tests_files", []))

    decision = classify_command(command, working_directory, tools.root)
    sig = command_signature(decision.argv)
    auto = state.get("auto_approve_safe", False)
    interactive = state.get("interactive", False)
    already_approved = sig in state.get("approved_commands", [])
    ask = needs_approval(decision, auto) and not already_approved

    new_logs = ["Running tests..."]
    new_tool_calls = [{"tool": tool_name, "command": command, "working_directory": working_directory, "reason": reason, "risk": decision.risk.value}]
    new_chat: List[Dict[str, Any]] = [_forge_msg("I need to run the test suite. This may require your approval.")]
    approved_commands_delta: List[str] = []
    approval_status = "not_required"

    if decision.risk == CommandRisk.BLOCKED:
        approved = False
        approval_status = "blocked"
    elif not ask:
        approved = True
        approval_status = "auto_approved" if already_approved or (decision.risk == CommandRisk.SAFE and auto) else "safe"
    else:
        if interactive and HITL_AVAILABLE:
            pending = {
                "command": command,
                "working_directory": working_directory,
                "reason": reason,
                "risk": decision.risk.value,
                "tool": tool_name,
                "signature": sig,
            }
            # Pause the graph; resumes with the user's decision dict.
            approval = interrupt(pending)  # type: ignore[misc]
            approved = bool(approval.get("approved"))
            scope = approval.get("scope", "once")
            if approved and scope == "session":
                approved_commands_delta.append(sig)
            approval_status = "approved" if approved else "denied"
        else:
            # Non-interactive (FastAPI) or no HITL: cannot ask -> skip.
            approved = False
            approval_status = "skipped_non_interactive"

    result = tools.run_command(command, working_directory=working_directory, reason=reason, approved=approved)

    card = {
        "kind": "terminal",
        "command": command,
        "working_directory": working_directory,
        "reason": reason,
        "approval": approval_status,
        "executed": result.get("executed", False),
        "blocked": result.get("blocked", False),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
        "exit_code": result.get("exit_code", -1),
        "duration_ms": result.get("duration_ms", 0),
        "timeout": result.get("timeout", False),
    }
    new_chat.append(_forge_msg("", tool_card=card))

    executed = result.get("executed", False)
    exit_code = result.get("exit_code", -1)
    if not executed:
        final = "skipped"
        report = result.get("stderr", "") or "Tests were not executed."
        new_logs.append(f"Tests not executed ({approval_status}).")
    else:
        passed = exit_code in (0, 5)  # 5 == no tests collected (not a failure)
        report = (result.get("stdout", "") or "") + (result.get("stderr", "") or "")
        if passed:
            final = "passed"
            new_logs.append("Tests passed.")
        else:
            final = "failed"
            new_logs.append(f"Tests failed (exit {exit_code}, iteration {state.get('iteration_count', 0)}).")

    return {
        "status": "running_tests",
        "current_agent": "Tester",
        "current_tool": tool_name,
        "test_results": report,
        "errors": "" if final == "passed" else report,
        "final_status": final,
        "requested_command": command,
        "command_output": result.get("stdout", ""),
        "command_error": result.get("stderr", ""),
        "command_exit_code": exit_code,
        "command_timeout": result.get("timeout", False),
        "approval_status": approval_status,
        "pending_approval": {},
        "logs": new_logs,
        "agent_messages": [_activity_entry("Tester", "Ran tests", "coding")],
        "usage_info": _bump_usage(state),
        "tool_calls": new_tool_calls,
        "tool_results": [card],
        "chat_messages": new_chat,
        "approved_commands": approved_commands_delta,
    }


def route_after_tests(state: DevState) -> str:
    """Continue to the debugger only while tests fail and we have budget."""
    if state.get("final_status") in ("passed", "skipped"):
        return "reviewer"
    if state.get("iteration_count", 0) >= MAX_DEBUG_ITERATIONS:
        return "reviewer"
    # Also check the LLM call budget to prevent runaway costs.
    llm_calls = state.get("usage_info", {}).get("llm_calls", 0)
    if llm_calls >= MAX_LLM_CALLS:
        return "reviewer"
    return "debugger"


async def node_debugger(state: DevState):
    """Fix code based on captured errors, then loop back to run_tests."""
    it = state.get("iteration_count", 0) + 1
    feedback = state.get("errors", "") or state.get("review_comments", "")
    fixed = await debugger_agent(state.get("code_files", []), feedback)
    return {
        "code_files": fixed,
        "files_modified": [f.get("path", "") for f in fixed],
        "status": "debugging",
        "current_agent": "Debugger",
        "iteration_count": it,
        "logs": [f"Debugging (attempt {it}/{MAX_AGENT_ITERATIONS})..."],
        "agent_messages": [_activity_entry("Debugger", f"Applied fixes (attempt {it})", "debugging")],
        "usage_info": _bump_usage(state),
        "chat_messages": [_forge_msg(f"Tests failed. I'm applying fixes (attempt {it}/{MAX_AGENT_ITERATIONS}).")],
    }


async def node_reviewer(state: DevState):
    review = await reviewer_agent(state["code_files"])
    return {
        "review_comments": review,
        "status": "reviewing",
        "current_agent": "Reviewer",
        "logs": ["Reviewing code..."],
        "agent_messages": [_activity_entry("Reviewer", "Review complete", "review")],
        "usage_info": _bump_usage(state),
        "chat_messages": [_forge_msg("Reviewing the code for quality and best practices.")],
    }


async def node_documentation(state: DevState):
    docs = await documentation_agent(state["code_files"], state["requirements"])
    return {
        "docs_files": docs,
        "status": "documenting",
        "current_agent": "Documentation",
        "logs": ["Writing documentation..."],
        "agent_messages": [_activity_entry("Documentation", f"Generated {len(docs)} doc files", "documentation")],
        "usage_info": _bump_usage(state),
        "chat_messages": [_forge_msg(f"Writing documentation ({len(docs)} file(s)).")],
    }


async def node_packager(state: DevState):
    await packager_agent(state)
    files_count = (
        len(state.get("code_files", []))
        + len(state.get("tests_files", []))
        + len(state.get("docs_files", []))
    )
    test_status = state.get("final_status", "unknown")
    iters = state.get("iteration_count", 0)
    # One light LLM call for Forge's wrap-up (falls back to deterministic).
    summary = await forge_summarize(state.get("prompt", ""), files_count, test_status, iters)
    return {
        "status": "completed",
        "final_status": "completed",
        "current_agent": "Packager",
        "logs": ["Packaging project...", "Project packaged."],
        "agent_messages": [_activity_entry("Forge", "Final summary", "fast")],
        "usage_info": _bump_usage(state),
        "chat_messages": [_forge_msg(summary)],
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(DevState)
    builder.add_node("understand", node_understand)
    builder.add_node("requirements", node_requirements)
    builder.add_node("architecture", node_architecture)
    builder.add_node("coder", node_coder)
    builder.add_node("validate", node_validate)
    builder.add_node("tester", node_tester)
    builder.add_node("run_tests", node_run_tests)
    builder.add_node("debugger", node_debugger)
    builder.add_node("reviewer", node_reviewer)
    builder.add_node("documentation", node_documentation)
    builder.add_node("packager", node_packager)

    builder.set_entry_point("understand")
    builder.add_edge("understand", "requirements")
    builder.add_edge("requirements", "architecture")
    builder.add_edge("architecture", "coder")
    builder.add_edge("coder", "validate")
    builder.add_edge("validate", "tester")
    builder.add_edge("tester", "run_tests")
    # Iterative debug loop (bounded by MAX_AGENT_ITERATIONS via route_after_tests)
    builder.add_conditional_edges(
        "run_tests",
        route_after_tests,
        {"reviewer": "reviewer", "debugger": "debugger"},
    )
    builder.add_edge("debugger", "run_tests")
    builder.add_edge("reviewer", "documentation")
    builder.add_edge("documentation", "packager")
    builder.add_edge("packager", END)

    compile_kwargs: Dict[str, Any] = {}
    if HITL_AVAILABLE:
        compile_kwargs["checkpointer"] = MemorySaver()
    return builder.compile(**compile_kwargs)


workflow = build_graph()
