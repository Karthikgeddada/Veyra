"""Centralized LLM configuration and factory for DevTeam AI.

All agents obtain their LLM through :func:`generate_response` (or
:func:`stream_response` for streaming), which delegates to the active
:class:`~config.providers.LLMProvider`. The provider is selected via
the ``LLM_PROVIDER`` environment variable (``nvidia``, ``openrouter``)
with auto-detection when unset.

A lightweight model-routing layer lets different task types (planning,
coding, debugging, review, documentation, fast, requirements,
architecture, testing, chat) use different models, each configurable
via environment variables with a safe fallback to the default model.

The legacy :func:`get_llm` factory (LangChain ``ChatOpenAI``) is preserved
for backward compatibility and used only when a LangChain object is
explicitly needed (e.g. some LangGraph integrations).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Generator, List, Optional, Union

from dotenv import load_dotenv

from config.providers import (
    LLMProvider,
    get_provider,
    get_provider_info,
    list_providers,
    provider_health_check,
)

load_dotenv()

logger = logging.getLogger("devteam_ai.llm")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Bounded execution limits
# ---------------------------------------------------------------------------
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "3"))
MAX_LLM_CALLS = int(os.getenv("MAX_LLM_CALLS", "50"))
MAX_COMMAND_EXECUTIONS = int(os.getenv("MAX_COMMAND_EXECUTIONS", "20"))
MAX_DEBUG_ITERATIONS = int(os.getenv("MAX_DEBUG_ITERATIONS", str(MAX_AGENT_ITERATIONS)))

# ---------------------------------------------------------------------------
# Active provider
# ---------------------------------------------------------------------------
_provider: Optional[LLMProvider] = None


def _get_active_provider() -> LLMProvider:
    """Return the cached active provider instance."""
    global _provider
    if _provider is None:
        _provider = get_provider()
    return _provider


def get_provider_info_safe() -> Dict[str, Any]:
    """Return info for the active provider (no secrets)."""
    return get_provider_info()


def health_check() -> Dict[str, Any]:
    """Run a health check on the active provider (no secrets, no LLM call)."""
    return provider_health_check()


# ---------------------------------------------------------------------------
# Default model — resolved from the active provider
# ---------------------------------------------------------------------------
def _resolve_default_model() -> str:
    return _get_active_provider().config.default_model


# Keep OPENROUTER_BASE_URL for backward compatibility.
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# DEFAULT_MODEL is now lazy (resolved from provider); keep a module-level
# accessor for backward compat.
DEFAULT_MODEL = _resolve_default_model()

# ---------------------------------------------------------------------------
# Task-specific model routing and parameter configuration
# ---------------------------------------------------------------------------
# Maps a logical task type to an environment-configured model name. Values
# that are unset / empty fall back to the provider's default model.
_TASK_ENV_VARS: Dict[str, str] = {
    "requirements": "LLM_MODEL_REQUIREMENTS",
    "planning": "LLM_MODEL_PLANNING",
    "architecture": "LLM_MODEL_ARCHITECTURE",
    "coding": "LLM_MODEL_CODING",
    "debugging": "LLM_MODEL_DEBUGGING",
    "testing": "LLM_MODEL_TESTING",
    "review": "LLM_MODEL_REVIEW",
    "documentation": "LLM_MODEL_DOCUMENTATION",
    "chat": "LLM_MODEL_CHAT",
    "fast": "LLM_MODEL_FAST",
}

# Per-task default parameters. These can be overridden by the caller, but
# provide sensible limits so a simple classification doesn't burn 8k tokens.
TASK_MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "requirements": {"max_tokens": 4096, "temperature": 0.3},
    "planning": {"max_tokens": 8192, "temperature": 0.3},
    "architecture": {"max_tokens": 8192, "temperature": 0.3},
    "coding": {"max_tokens": 8192, "temperature": 0.2},
    "debugging": {"max_tokens": 4096, "temperature": 0.2},
    "testing": {"max_tokens": 4096, "temperature": 0.2},
    "review": {"max_tokens": 4096, "temperature": 0.3},
    "documentation": {"max_tokens": 4096, "temperature": 0.3},
    "chat": {"max_tokens": 2048, "temperature": 0.4},
    "fast": {"max_tokens": 256, "temperature": 0.2},
}

# Cache of resolved models.
_resolved_models: Dict[str, str] = {}


def get_model_for_task(task_type: Optional[str] = None) -> str:
    """Return the configured model for a task type, falling back to default."""
    default_model = _resolve_default_model()
    if not task_type:
        return default_model
    if task_type in _resolved_models:
        return _resolved_models[task_type]
    env_var = _TASK_ENV_VARS.get(task_type, "LLM_MODEL")
    model = os.getenv(env_var) or os.getenv("LLM_MODEL") or default_model
    _resolved_models[task_type] = model
    return model


def get_task_config(task_type: Optional[str] = None) -> Dict[str, Any]:
    """Return the task-specific config (max_tokens, temperature) for a task."""
    if not task_type or task_type not in TASK_MODEL_CONFIG:
        return {"max_tokens": 4096, "temperature": 0.2}
    return dict(TASK_MODEL_CONFIG[task_type])


def list_task_models() -> Dict[str, str]:
    """Return a dict of task_type -> resolved model (for UI display)."""
    default_model = _resolve_default_model()
    result = {"default": default_model}
    for task in _TASK_ENV_VARS:
        result[task] = get_model_for_task(task)
    return result


# ---------------------------------------------------------------------------
# API key access (backward compatible — delegates to the active provider)
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    """Return the active provider's API key, raising if missing.

    Kept for backward compatibility with code that needs the raw key
    (e.g. error redaction in legacy paths). New code should use the
    provider abstraction directly.
    """
    cfg = _get_active_provider().config
    if not cfg.api_key:
        raise ValueError(
            f"{cfg.name.upper()}_API_KEY is not set. Configure it in your "
            f".env file or Streamlit Secrets. See the README for setup instructions."
        )
    return cfg.api_key


# ---------------------------------------------------------------------------
# Legacy LangChain factory (preserved for backward compat)
# ---------------------------------------------------------------------------
def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.2,
    task_type: Optional[str] = None,
    max_tokens: int = 4096,
    timeout: Optional[int] = 120,
):
    """Build and return a LangChain ``ChatOpenAI`` for the active provider.

    This is the legacy LangChain-based factory. New code should prefer
    :func:`generate_response` / :func:`stream_response` which use the
    provider abstraction directly (OpenAI SDK, no LangChain dependency).

    Parameters
    ----------
    model:
        Explicit model id. If omitted, the model is resolved from
        ``task_type`` via the routing layer, falling back to the default.
    temperature:
        Sampling temperature passed through to the provider.
    task_type:
        Logical task (planning, coding, debugging, review, etc.).
    max_tokens:
        Maximum tokens for the completion.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    ChatOpenAI
        A ready-to-use LangChain chat model.
    """
    from langchain_openai import ChatOpenAI

    provider = _get_active_provider()
    cfg = provider.config
    chosen_model = model or get_model_for_task(task_type)
    return ChatOpenAI(
        model=chosen_model,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=cfg.api_key,
        openai_api_base=cfg.base_url,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Primary generation helpers (provider-agnostic, OpenAI SDK)
# ---------------------------------------------------------------------------
async def generate_response(
    prompt_or_messages: Union[str, List[Dict[str, str]]],
    task_type: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a single completion from a prompt or chat message list.

    This delegates to the active provider's ``generate()`` method (OpenAI
    SDK, no LangChain dependency). Both plain strings and OpenAI-style
    message dicts are accepted.

    When ``task_type`` is provided, the task-specific config (max_tokens,
    temperature) is used as defaults; explicit parameters override them.
    The model is resolved from the task routing layer.
    """
    provider = _get_active_provider()
    task_cfg = get_task_config(task_type)

    effective_model = model or get_model_for_task(task_type)
    effective_temp = temperature if temperature is not None else task_cfg["temperature"]
    effective_max = max_tokens if max_tokens is not None else task_cfg["max_tokens"]

    try:
        return provider.generate(
            prompt_or_messages,
            model=effective_model,
            temperature=effective_temp,
            max_tokens=effective_max,
        )
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM generate_response failed: %s", exc)
        raise


def stream_response(
    prompt_or_messages: Union[str, List[Dict[str, str]]],
    task_type: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Generator[str, None, None]:
    """Stream a completion as text chunks from the active provider.

    Yields only the final answer text (reasoning_content is consumed but
    never yielded). Useful for the Streamlit UI to display Forge's
    response progressively.
    """
    provider = _get_active_provider()
    task_cfg = get_task_config(task_type)

    effective_model = model or get_model_for_task(task_type)
    effective_temp = temperature if temperature is not None else task_cfg["temperature"]
    effective_max = max_tokens if max_tokens is not None else task_cfg["max_tokens"]

    yield from provider.stream(
        prompt_or_messages,
        model=effective_model,
        temperature=effective_temp,
        max_tokens=effective_max,
    )
