"""Project generation service for the FastAPI backend.

Keeps an in-memory registry of active workflow runs and exposes simple
helpers used by the API routes. The heavy lifting lives in the LangGraph
workflow; this module just owns run lifecycle and state access.

FastAPI runs are *non-interactive*: safe commands (pytest, etc.) auto-run
and approval-required commands are skipped (no human to ask), so the
graph never blocks on an interrupt.
"""

import asyncio
import uuid
from typing import Dict

from workflows.graph import DevState, make_config, workflow


active_runs: Dict[str, DevState] = {}


def _initial_state(run_id: str, prompt: str) -> DevState:
    return {
        "run_id": run_id,
        "prompt": prompt,
        "requirements": "",
        "architecture": "",
        "code_files": [],
        "review_comments": "",
        "tests_files": [],
        "docs_files": [],
        "status": "starting",
        "logs": [],
        "current_agent": "",
        "agent_messages": [],
        "test_results": "",
        "errors": "",
        "iteration_count": 0,
        "final_status": "",
        "project_spec": "",
        "files_modified": [],
        "usage_info": {"agent_calls": 0, "models": {}, "max_iterations": 3},
        # Forge interactive state (non-interactive for the API backend)
        "interactive": False,
        "auto_approve_safe": True,
        "chat_messages": [],
        "tool_calls": [],
        "tool_results": [],
        "pending_approval": {},
        "approval_status": "",
        "requested_command": "",
        "command_output": "",
        "command_error": "",
        "command_exit_code": -1,
        "command_timeout": False,
        "current_tool": "",
        "approved_commands": [],
        # Forge bounded execution + project understanding
        "llm_calls": 0,
        "command_executions": 0,
        "project_analysis": {},
    }


async def start_generation(prompt: str) -> str:
    """Start a new generation run in the background and return its id."""
    run_id = str(uuid.uuid4())
    state = _initial_state(run_id, prompt)
    active_runs[run_id] = state
    asyncio.create_task(run_graph(run_id, state))
    return run_id


async def run_graph(run_id: str, state: DevState) -> None:
    """Drive the LangGraph workflow to completion, capturing errors.

    Uses a per-run thread_id so the MemorySaver checkpointer can track state.
    Non-interactive runs never pause on an interrupt.
    """
    config = make_config(run_id)
    try:
        async for output in workflow.astream(state, config=config, stream_mode="updates"):
            for _node_name, node_output in output.items():
                active_runs[run_id].update(node_output)
        # Mirror the authoritative checkpointed state back into the registry.
        try:
            snapshot = workflow.get_state(config)
            if snapshot and getattr(snapshot, "values", None):
                active_runs[run_id].update(dict(snapshot.values))
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001 - record and surface to the client
        active_runs[run_id]["status"] = "failed"
        active_runs[run_id]["final_status"] = "failed"
        active_runs[run_id]["logs"].append(f"Error: {exc}")
        active_runs[run_id]["errors"] = str(exc)


def get_status(run_id: str):
    return active_runs.get(run_id)


def get_files(run_id: str):
    """Return all generated files (code + tests + docs) for a run."""
    state = active_runs.get(run_id)
    if not state:
        return []
    all_files = []
    all_files.extend(state.get("code_files", []))
    all_files.extend(state.get("tests_files", []))
    all_files.extend(state.get("docs_files", []))
    return all_files
