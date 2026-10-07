"""DevTeam AI - Forge: Interactive AI Software Engineering Workspace.

A two-panel Streamlit workspace:
  * LEFT  - project / file explorer + code viewer
  * RIGHT - Forge conversation (user / Forge / tool cards / approval)

Forge drives a LangGraph workflow that can run commands through a
controlled tool layer. Potentially risky commands pause the graph
(LangGraph ``interrupt``) and surface an interactive approval card in the
chat. After the user decides (allow once / allow session / deny), the graph
resumes via ``Command(resume=...)`` and Forge continues.

There is no permanent bottom terminal: command execution appears as a
terminal card *inside* the Forge conversation.
"""

import asyncio
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from config.llm_config import MAX_AGENT_ITERATIONS, health_check, list_task_models
from workflows.graph import HITL_AVAILABLE, DevState, make_config, workflow
from config.providers import get_provider_info

try:  # Command is only needed to resume an interrupted graph.
    from langgraph.types import Command
except Exception:  # noqa: BLE001
    Command = None  # type: ignore[assignment]

load_dotenv()

st.set_page_config(page_title="DevTeam AI - Forge", page_icon="🛠️", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .forge-title { font-size:1.6rem; font-weight:800; letter-spacing:-0.02em; }
        .forge-sub { color:#94a3b8; font-size:0.95rem; }
        .agent-pill {
            display:inline-block; padding:3px 10px; margin:2px 3px;
            border-radius:12px; font-size:0.75rem; font-weight:600;
            border:1px solid #444; background:#1e1e1e; color:#aaa;
        }
        .agent-pill.active { background:#2563eb; border-color:#3b82f6; color:#fff; }
        .agent-pill.done { background:#14532d; border-color:#22c55e; color:#bbf7d0; }
        .agent-pill.failed { background:#7f1d1d; border-color:#ef4444; color:#fecaca; }
        .term-card { background:#0b0f14; color:#d4d4d4; border:1px solid #222;
            border-radius:8px; padding:10px 12px; font-family: monospace; font-size:0.78rem; }
        .bubble-user { background:#1e293b; border:1px solid #334155; border-radius:10px;
            padding:8px 12px; margin:6px 0; }
        .bubble-forge { background:#0f172a; border:1px solid #1e3a8a; border-radius:10px;
            padding:8px 12px; margin:6px 0; }
        .stat-card { background:#111827; padding:8px 12px; border-radius:8px; border:1px solid #1f2937; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("<span class='forge-title'>🛠️ DEVTEAM AI</span>", unsafe_allow_html=True)
st.markdown("<span class='forge-sub'>Forge — AI Software Engineer</span>", unsafe_allow_html=True)

# Provider / model status badge
_provider_display = _provider_name.upper()
_model_display = _provider_model.split("/")[-1] if _provider_model and _provider_model != "unknown" else "unknown"
st.markdown(
    f"<span class='forge-sub'>Provider: **{_provider_display}** · Model: **{_model_display}** · Status: **✓ Configured**</span>",
    unsafe_allow_html=True,
)

os.makedirs("generated_projects", exist_ok=True)

# ---------------------------------------------------------------------------
# Provider status guard (NVIDIA / OpenRouter — provider-agnostic)
# ---------------------------------------------------------------------------
_provider_info = get_provider_info()
_provider_name = _provider_info.get("provider", "unknown")
_provider_model = _provider_info.get("model", "unknown")
_provider_configured = _provider_info.get("configured", False)

if not _provider_configured:
    _key_name = "NVIDIA_API_KEY" if _provider_name == "nvidia" else "OPENROUTER_API_KEY"
    st.error(
        f"**{_key_name} is not set.** Add it to your `.env` file or Streamlit "
        f"Secrets (copy `.env.example` to `.env`)."
    )
    st.stop()

if not HITL_AVAILABLE:
    st.warning(
        "Human-in-the-loop approval requires `langgraph>=0.2.31`. The installed "
        "version lacks `interrupt`; Forge will run non-interactively (safe commands "
        "auto-run, others skipped). Upgrade langgraph to enable approval cards."
    )

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("forge_state", {})
ss.setdefault("pending_approval", None)
ss.setdefault("forge_config", None)
ss.setdefault("forge_initial", None)
ss.setdefault("auto_approve_safe", False)

AGENTS = ["Project Analyzer", "Requirement Analyzer", "Architect", "Developer", "Validator", "Tester", "Debugger", "Reviewer", "Documentation", "Packager"]


def _build_initial_state(prompt: str, run_id: str, auto_approve_safe: bool) -> DevState:
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
        "usage_info": {"agent_calls": 0, "models": {}, "max_iterations": MAX_AGENT_ITERATIONS},
        "interactive": True,
        "auto_approve_safe": auto_approve_safe,
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


# ---------------------------------------------------------------------------
# Graph streaming + interrupt/resume
# ---------------------------------------------------------------------------
async def _stream_chunk(resume_value=None):
    """Stream the graph until it pauses (interrupt) or finishes; return state."""
    cfg = ss["forge_config"]
    if resume_value is not None:
        gen = workflow.astream(Command(resume=resume_value), config=cfg, stream_mode="updates")
    else:
        gen = workflow.astream(ss["forge_initial"], config=cfg, stream_mode="updates")
    async for _output in gen:
        pass  # state is tracked by the MemorySaver checkpointer
    return workflow.get_state(cfg)


def _extract_pending(snap) -> dict | None:
    """Pull the interrupt payload from a paused StateSnapshot, defensively."""
    if not snap or not getattr(snap, "next", None):
        return None
    for task in getattr(snap, "tasks", []) or []:
        interrupts = getattr(task, "interrupts", None) or []
        for it in interrupts:
            value = getattr(it, "value", None)
            if value is None and isinstance(it, tuple) and len(it) >= 2:
                value = it[1]
            if value:
                return value
    return None


def _apply_snapshot(snap) -> None:
    vals = dict(snap.values) if snap and getattr(snap, "values", None) else {}
    ss["forge_state"] = vals
    ss["pending_approval"] = _extract_pending(snap)


def _run_and_apply(resume_value=None) -> None:
    try:
        snap = asyncio.run(_stream_chunk(resume_value=resume_value))
        _apply_snapshot(snap)
    except Exception as exc:  # noqa: BLE001
        ss["pending_approval"] = None
        cur = dict(ss.get("forge_state") or {})
        cur["status"] = "failed"
        cur["final_status"] = "failed"
        cur["errors"] = str(exc)
        cur.setdefault("chat_messages", []).append({"role": "forge", "agent": "Forge", "content": f"I hit an error: {exc}"})
        ss["forge_state"] = cur
    st.rerun()


def _resume(decision: dict) -> None:
    _run_and_apply(resume_value=decision)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def agent_pills(state: dict) -> None:
    current = state.get("current_agent", "")
    done = {m["agent"] for m in state.get("agent_messages", [])}
    failed = state.get("status") == "failed"
    html = []
    for a in AGENTS:
        if failed and a == current:
            cls = "failed"
        elif a in done:
            cls = "done"
        elif a == current:
            cls = "active"
        else:
            cls = ""
        mark = "✓" if a in done else ("●" if a == current else "○")
        html.append(f"<span class='agent-pill {cls}'>{mark} {a}</span>")
    st.markdown("  ".join(html), unsafe_allow_html=True)


def file_icon(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {".py": "🐍", ".js": "📜", ".ts": "🟦", ".html": "🌐", ".css": "🎨",
            ".md": "📝", ".json": "🔧", ".txt": "📄", ".sql": "🗄️", ".yml": "⚙️",
            ".yaml": "⚙️", ".env": "🔑"}.get(ext, "📄")


def lang_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {".py": "python", ".js": "javascript", ".ts": "typescript", ".html": "html",
            ".css": "css", ".md": "markdown", ".json": "json", ".sql": "sql",
            ".sh": "bash"}.get(ext, "plaintext")


def render_terminal_card(card: dict) -> None:
    """Render a compact terminal tool card inside the conversation."""
    executed = card.get("executed", False)
    exit_code = card.get("exit_code", -1)
    if not executed:
        status_badge = "⏸ not executed"
    elif exit_code in (0, 5):
        status_badge = "✓ Done"
    else:
        status_badge = "✗ Failed"
    with st.container(border=True):
        st.markdown(f"**⚙ Terminal** &nbsp; `{status_badge}`")
        st.code(f"$ {card.get('command', '')}", language="bash")
        if card.get("reason"):
            st.caption(f"Reason: {card['reason']}")
        meta = []
        if card.get("approval") and card["approval"] != "not_required":
            meta.append(f"Approval: {card['approval']}")
        meta.append(f"Exit code: {exit_code}")
        meta.append(f"Duration: {card.get('duration_ms', 0)} ms")
        if card.get("timeout"):
            meta.append("TIMEOUT")
        st.caption(" · ".join(meta))
        out = card.get("stdout", "") or ""
        err = card.get("stderr", "") or ""
        if out:
            with st.expander("stdout", expanded=False):
                st.code(out[-4000:], language="text")
        if err:
            with st.expander("stderr", expanded=False):
                st.code(err[-4000:], language="text")


def render_approval_card(pending: dict) -> None:
    """Interactive approval request (allow once / allow session / deny)."""
    with st.container(border=True):
        st.warning(f"⚠️ Forge wants to run:  `$ {pending.get('command', '')}`")
        if pending.get("reason"):
            st.caption(f"Reason: {pending['reason']}")
        st.caption(f"Risk: {pending.get('risk', '?')}  ·  Working dir: {pending.get('working_directory', '.')}")
        c1, c2, c3 = st.columns(3)
        if c1.button("Allow once", key="alw_once", type="primary"):
            _resume({"approved": True, "scope": "once"})
        if c2.button("Allow session", key="alw_sess"):
            _resume({"approved": True, "scope": "session"})
        if c3.button("Deny", key="deny_cmd"):
            _resume({"approved": False, "scope": "deny"})


def render_chat(state: dict) -> None:
    """Render the Forge conversation from chat_messages."""
    messages = state.get("chat_messages", [])
    if not messages and not ss.get("pending_approval"):
        st.markdown("_<span class='forge-sub'>Send a prompt to start working with Forge.</span>_", unsafe_allow_html=True)
        return
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            st.markdown(f"<div class='bubble-user'><b>You:</b> {msg.get('content', '')}</div>", unsafe_allow_html=True)
        elif role == "forge":
            card = msg.get("tool_card")
            if card:
                render_terminal_card(card)
            else:
                st.markdown(f"<div class='bubble-forge'><b>Forge:</b> {msg.get('content', '')}</div>", unsafe_allow_html=True)
    # Approval card (if paused) appears at the end of the conversation.
    if ss.get("pending_approval"):
        render_approval_card(ss["pending_approval"])


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
state = ss.get("forge_state") or {}

# Metrics + model header
models = list_task_models()
_usage = state.get("usage_info", {})
mcol = st.columns([1, 1, 1, 1, 1])
mcol[0].metric("Status", (state.get("status") or "idle").title())
mcol[1].metric("Agent calls", _usage.get("agent_calls", 0))
mcol[2].metric("LLM calls", _usage.get("llm_calls", 0))
mcol[3].metric("Debug iterations", state.get("iteration_count", 0))
files_count = len(state.get("code_files", [])) + len(state.get("tests_files", [])) + len(state.get("docs_files", []))
mcol[4].metric("Files", files_count)
with st.expander("Model routing", expanded=False):
    st.markdown(f"Provider: `{_provider_name}` | Model: `{_model_display}`")
    st.markdown("  ".join(f"`{t}: {m.split('/')[-1]}`" for t, m in models.items()))

# Agent activity pills (from real state)
agent_pills(state)

left, right = st.columns([1, 2], gap="large")

# ---- LEFT: project / file explorer ----
with left:
    st.subheader("📁 Project")
    all_files = []
    all_files.extend(state.get("code_files", []))
    all_files.extend(state.get("tests_files", []))
    all_files.extend(state.get("docs_files", []))
    if not all_files:
        st.info("No files yet. They will appear here as Forge generates them.")
    else:
        file_paths = [f.get("path") for f in all_files if f.get("path")]
        labeled = [f"{file_icon(p)}  {p}" for p in file_paths]
        choice = st.radio("Explorer", labeled, index=0, label_visibility="collapsed", key="file_explorer")
        selected = file_paths[labeled.index(choice)]
        st.markdown(f"**Selected:** `{selected}`")
        content = ""
        for f in all_files:
            if f.get("path") == selected:
                content = f.get("content", "")
                break
        st.code(content, language=lang_for(selected))
        run_id = state.get("run_id")
        zip_path = f"generated_projects/{run_id}.zip" if run_id else None
        if zip_path and os.path.exists(zip_path):
            with open(zip_path, "rb") as fobj:
                st.download_button("Download Project (ZIP) 📦", fobj, file_name=f"devteam_project_{run_id}.zip", mime="application/zip")

# ---- RIGHT: Forge conversation ----
with right:
    st.subheader("🤖 Forge")
    st.caption("AI Software Engineer")
    render_chat(state)
    # Collapsed agent logs (no permanent terminal)
    logs = state.get("logs", [])
    if logs:
        with st.expander("Agent logs", expanded=False):
            st.code("\n".join(f"> {l}" for l in logs[-200:]), language="bash")

    st.divider()
    # Prompt input + controls
    ap_col, new_col = st.columns([3, 1])
    ap_col.checkbox("Auto-approve safe commands (pytest, git status…)", key="auto_approve_safe", help="If checked, safe commands run without an approval card. Leave unchecked to see the approval flow.")
    if new_col.button("🔄 New project", help="Clear the current session and start over."):
        for k in ("forge_state", "pending_approval", "forge_config", "forge_initial"):
            ss.pop(k, None)
        ss["forge_state"] = {}
        ss["pending_approval"] = None
        st.rerun()

    prompt = st.text_area(
        "What do you want Forge to build?",
        height=80,
        placeholder="E.g., Create a simple Python calculator module with tests. Run the tests and fix any failures.",
        key="forge_prompt",
    )
    if st.button("Send to Forge ⚡", type="primary"):
        txt = (prompt or "").strip()
        if not txt:
            st.warning("Please enter a prompt to begin.")
        else:
            run_id = str(uuid.uuid4())
            ss["forge_config"] = make_config(run_id)
            ss["forge_initial"] = _build_initial_state(txt, run_id, bool(ss.get("auto_approve_safe", False)))
            ss["pending_approval"] = None
            with st.spinner("Forge is working…"):
                _run_and_apply()


