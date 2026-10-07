"""Forge - the AI software engineer voice for DevTeam AI.

Most of Forge's narration is produced deterministically by the workflow
nodes (to keep LLM cost down), but a single short LLM call is used at the
end of a run to produce a human-style wrap-up message. If that call fails,
Forge falls back to a deterministic summary, so the workflow never breaks.

Forge also supports streaming responses via :func:`stream_forge_message`
which uses the provider abstraction to yield text chunks progressively.
"""

from __future__ import annotations

import logging
from typing import Generator, Optional

from config.llm_config import generate_response, stream_response

logger = logging.getLogger("devteam_ai.forge")

# Reasoning indicators shown in the UI while Forge is "thinking".
REASONING_INDICATORS = {
    "planning": "Forge is planning...",
    "requirements": "Forge is analyzing requirements...",
    "architecture": "Forge is designing the architecture...",
    "coding": "Forge is writing code...",
    "debugging": "Forge is analyzing the test failure...",
    "testing": "Forge is running tests...",
    "review": "Forge is reviewing the code...",
    "documentation": "Forge is writing documentation...",
    "fast": "Forge is reasoning...",
    "chat": "Forge is thinking...",
}


def get_reasoning_indicator(task_type: Optional[str] = None) -> str:
    """Return a human-friendly status message for a task type."""
    if not task_type:
        return "Forge is reasoning..."
    return REASONING_INDICATORS.get(task_type, "Forge is working...")


def deterministic_summary(prompt: str, files_count: int, test_status: str, iterations: int) -> str:
    """A no-LLM Forge wrap-up used as a safe fallback."""
    status_word = "passing" if test_status == "passed" else "best-effort"
    return (
        f"I've finished working on your request: \"{prompt[:120]}\". "
        f"I generated {files_count} file(s), ran the tests, and reached {status_word} "
        f"results after {iterations} debug iteration(s). You can browse the files "
        f"on the left and download the project."
    )


async def forge_summarize(prompt: str, files_count: int, test_status: str, iterations: int) -> str:
    """Produce a short Forge wrap-up message (one LLM call, 'fast' model).

    Falls back to :func:`deterministic_summary` on any error so the workflow
    is resilient to LLM failures.
    """
    summary_prompt = (
        "You are Forge, an AI software engineer. In 2-3 sentences, summarize the "
        "work you just completed for the user in a friendly, professional tone. "
        f"User request: {prompt[:300]}. Files generated: {files_count}. "
        f"Test status: {test_status}. Debug iterations: {iterations}. "
        "Do not use markdown headings."
    )
    try:
        return await generate_response(summary_prompt, task_type="fast", max_tokens=256)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Forge summary LLM call failed (%s); using deterministic summary.", exc)
        return deterministic_summary(prompt, files_count, test_status, iterations)


def stream_forge_message(
    prompt: str,
    context: Optional[str] = None,
    task_type: str = "chat",
) -> Generator[str, None, None]:
    """Stream a Forge chat response as text chunks.

    Yields only the final answer text (reasoning content is consumed by
    the provider but never yielded). Falls back to a single deterministic
    message if streaming fails.
    """
    full_prompt = prompt
    if context:
        full_prompt = f"{context}\n\nUser: {prompt}"
    try:
        yield from stream_response(full_prompt, task_type=task_type)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Forge stream failed (%s); sending fallback message.", exc)
        yield "I encountered an issue with the streaming connection. Please try again."