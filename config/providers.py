"""Provider abstraction for DevTeam AI LLM access.

All providers use the OpenAI Python SDK (``openai.OpenAI``) to talk to
OpenAI-compatible endpoints. The active provider is selected via the
``LLM_PROVIDER`` environment variable and exposes a uniform interface::

    provider.get_client()      -> OpenAI client
    provider.generate(...)     -> completion string (non-streaming)
    provider.stream(...)       -> generator of text chunks (streaming)
    provider.health_check()    -> dict with status info (no secrets)

Providers implemented:
    nvidia     - NVIDIA hosted endpoint (Nemotron models, reasoning support)
    openrouter - OpenRouter (preserved for backward compatibility)

Adding a new provider only requires subclassing :class:`LLMProvider` and
registering it in ``_PROVIDERS`` below.

Security: API keys are read from environment variables only and are never
logged, returned in health checks, or included in error messages.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("devteam_ai.providers")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Message type alias (OpenAI chat format)
# ---------------------------------------------------------------------------
Messages = Union[str, List[Dict[str, str]]]

# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------
_SECRET_REPLACEMENT = "[REDACTED]"


def _redact(text: str, secret: Optional[str] = None) -> str:
    """Remove API keys and secret-like patterns from *text*."""
    if not text:
        return text
    result = text
    if secret:
        result = result.replace(secret, _SECRET_REPLACEMENT)
    patterns = [
        (r"(sk-or-v1-)[A-Za-z0-9_-]+", r"\1" + _SECRET_REPLACEMENT),
        (r"(nvapi-)[A-Za-z0-9_-]+", r"\1" + _SECRET_REPLACEMENT),
        (r"(API_KEY=)\S+", r"\1" + _SECRET_REPLACEMENT),
        (r"(TOKEN=)\S+", r"\1" + _SECRET_REPLACEMENT),
        (r"(SECRET=)\S+", r"\1" + _SECRET_REPLACEMENT),
        (r"(PASSWORD=)\S+", r"\1" + _SECRET_REPLACEMENT),
    ]
    for pat, repl in patterns:
        result = re.sub(pat, repl, result)
    return result


# ---------------------------------------------------------------------------
# Provider configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class ProviderConfig:
    """Resolved configuration for a provider instance."""
    name: str
    base_url: str
    api_key: str
    default_model: str
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 4096
    timeout: int = 120
    extra_body: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base provider
# ---------------------------------------------------------------------------
class LLMProvider:
    """Base class for OpenAI-compatible LLM providers."""

    provider_name: str = "base"

    def __init__(self) -> None:
        self._config: Optional[ProviderConfig] = None
        self._client: Any = None

    @classmethod
    def _build_config(cls) -> ProviderConfig:
        """Read environment variables and return a resolved config."""
        raise NotImplementedError

    @property
    def config(self) -> ProviderConfig:
        if self._config is None:
            self._config = self._build_config()
        return self._config

    @property
    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    def get_client(self) -> Any:
        """Return a cached ``openai.OpenAI`` client (lazy import)."""
        if self._client is None:
            from openai import OpenAI
            cfg = self.config
            self._client = OpenAI(
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                timeout=cfg.timeout,
            )
        return self._client

    def _normalise_messages(self, messages: Messages) -> List[Dict[str, str]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return list(messages)

    def _build_request_kwargs(
        self,
        messages: Messages,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        cfg = self.config
        merged_extra = dict(cfg.extra_body)
        if extra_body:
            merged_extra.update(extra_body)
        kwargs: Dict[str, Any] = {
            "model": model or cfg.default_model,
            "messages": self._normalise_messages(messages),
            "temperature": temperature if temperature is not None else cfg.temperature,
            "max_tokens": max_tokens if max_tokens is not None else cfg.max_tokens,
            "stream": stream,
        }
        effective_top_p = top_p if top_p is not None else cfg.top_p
        if effective_top_p != 1.0:
            kwargs["top_p"] = effective_top_p
        if merged_extra:
            kwargs["extra_body"] = merged_extra
        return kwargs

    def generate(
        self,
        messages: Messages,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        retries: int = 3,
        backoff_base: float = 1.0,
    ) -> str:
        """Call chat-completions and return the text content.

        Retries with exponential backoff on transient errors. Auth errors
        are not retried. The API key is never in any raised exception.
        """
        from openai import (
            APIError, APIConnectionError, APITimeoutError,
            RateLimitError, AuthenticationError,
        )

        cfg = self.config
        if not cfg.api_key:
            raise RuntimeError(
                f"{self.provider_name} API key is not configured. "
                f"Set the environment variable and restart."
            )

        client = self.get_client()
        kwargs = self._build_request_kwargs(
            messages, model, temperature, max_tokens, top_p, extra_body, stream=False,
        )

        last_exc: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                start = time.time()
                response = client.chat.completions.create(**kwargs)
                duration = time.time() - start
                usage_tokens = "?"
                if response.usage:
                    usage_tokens = getattr(response.usage, "total_tokens", "?")
                
                # Extract content from the response
                if not response.choices or len(response.choices) == 0:
                    logger.error("LLM response contained no choices")
                    return ""
                
                message = response.choices[0].message
                content = getattr(message, "content", None) or ""
                
                # Log response metadata
                logger.info(
                    "LLM generate ok | provider=%s model=%s tokens=%s duration=%.2fs content_len=%d",
                    self.provider_name, kwargs["model"], usage_tokens, duration, len(content),
                )
                
                # Warn if content is empty
                if not content:
                    logger.warning(
                        "LLM returned empty content | provider=%s model=%s finish_reason=%s",
                        self.provider_name, kwargs["model"],
                        getattr(response.choices[0], "finish_reason", "unknown"),
                    )
                
                return content
            except AuthenticationError as exc:
                safe = _redact(str(exc), cfg.api_key)
                logger.error("LLM auth error: %s", safe)
                raise RuntimeError(
                    f"{self.provider_name} authentication failed. "
                    f"Check your API key configuration."
                ) from exc
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                last_exc = exc
                wait = backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    "LLM transient error (attempt %d/%d): %s -- retrying in %.1fs",
                    attempt, retries, _redact(str(exc), cfg.api_key), wait,
                )
                if attempt < retries:
                    time.sleep(wait)
            except APIError as exc:
                safe = _redact(str(exc), cfg.api_key)
                logger.error("LLM API error: %s", safe)
                raise RuntimeError(f"{self.provider_name} API error: {safe}") from exc
            except Exception as exc:  # noqa: BLE001
                safe = _redact(str(exc), cfg.api_key)
                logger.error("LLM unexpected error: %s", safe)
                raise RuntimeError(f"{self.provider_name} request failed: {safe}") from exc

        safe = _redact(str(last_exc), cfg.api_key) if last_exc else "unknown"
        raise RuntimeError(
            f"{self.provider_name} request failed after {retries} retries: {safe}"
        )

    def stream(
        self,
        messages: Messages,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Yield text chunks from a streaming chat-completion.

        Reasoning content (reasoning_content on the delta) is consumed but
        never yielded -- only the final answer text is streamed.
        """
        cfg = self.config
        if not cfg.api_key:
            raise RuntimeError(f"{self.provider_name} API key is not configured.")

        client = self.get_client()
        kwargs = self._build_request_kwargs(
            messages, model, temperature, max_tokens, top_p, extra_body, stream=True,
        )

        try:
            stream_resp = client.chat.completions.create(**kwargs)
            for chunk in stream_resp:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text
        except Exception as exc:  # noqa: BLE001
            safe = _redact(str(exc), cfg.api_key)
            logger.error("LLM stream error: %s", safe)
            raise RuntimeError(f"{self.provider_name} stream failed: {safe}") from exc

    def health_check(self) -> Dict[str, Any]:
        """Return a status dict without exposing secrets.

        Checks that the API key is set. Does not make an expensive LLM
        request -- at most a lightweight models.list() call.
        """
        cfg = self.config
        result: Dict[str, Any] = {
            "provider": self.provider_name,
            "model": cfg.default_model,
            "base_url": cfg.base_url,
            "configured": bool(cfg.api_key),
            "reachable": False,
            "error": "",
        }
        if not cfg.api_key:
            result["error"] = f"{self.provider_name} API key is not configured."
            return result
        try:
            client = self.get_client()
            client.models.list()
            result["reachable"] = True
        except Exception as exc:  # noqa: BLE001
            result["error"] = _redact(str(exc), cfg.api_key)
        return result

    def info(self) -> Dict[str, Any]:
        """Return public provider info (no secrets)."""
        cfg = self.config
        return {
            "provider": self.provider_name,
            "model": cfg.default_model,
            "base_url": cfg.base_url,
            "configured": bool(cfg.api_key),
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        }


# ---------------------------------------------------------------------------
# NVIDIA provider
# ---------------------------------------------------------------------------
class NVIDIAProvider(LLMProvider):
    """NVIDIA hosted OpenAI-compatible endpoint.

    Defaults to Nemotron 3 Ultra 550B with reasoning enabled.
    Configuration via environment variables:
        NVIDIA_API_KEY   - API key (required)
        NVIDIA_BASE_URL   - endpoint (default https://integrate.api.nvidia.com/v1)
        NVIDIA_MODEL      - model id (default nvidia/nemotron-3-ultra-550b-a55b)
    """

    provider_name = "nvidia"

    @classmethod
    def _build_config(cls) -> ProviderConfig:
        api_key = os.getenv("NVIDIA_API_KEY", "")
        if not api_key:
            try:
                import streamlit as st  # type: ignore
                api_key = st.secrets.get("NVIDIA_API_KEY", "")
            except Exception:
                pass
        return ProviderConfig(
            name="nvidia",
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=api_key,
            default_model=os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
            temperature=float(os.getenv("NVIDIA_TEMPERATURE", "0.2")),
            top_p=float(os.getenv("NVIDIA_TOP_P", "0.95")),
            max_tokens=int(os.getenv("NVIDIA_MAX_TOKENS", "4096")),
            timeout=int(os.getenv("NVIDIA_TIMEOUT", "120")),
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )


# ---------------------------------------------------------------------------
# OpenRouter provider (preserved for backward compatibility)
# ---------------------------------------------------------------------------
class OpenRouterProvider(LLMProvider):
    """OpenRouter OpenAI-compatible endpoint.

    Configuration via environment variables:
        OPENROUTER_API_KEY   - API key (required)
        OPENROUTER_BASE_URL   - endpoint (default https://openrouter.ai/api/v1)
        OPENROUTER_MODEL      - default model
    """

    provider_name = "openrouter"

    @classmethod
    def _build_config(cls) -> ProviderConfig:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            try:
                import streamlit as st  # type: ignore
                api_key = st.secrets.get("OPENROUTER_API_KEY", "")
            except Exception:
                pass
        default_model = os.getenv(
            "OPENROUTER_MODEL",
            os.getenv("OPENROUTER_MODEL_DEFAULT", "meta-llama/llama-3.3-70b-instruct:free"),
        )
        return ProviderConfig(
            name="openrouter",
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key,
            default_model=default_model,
            temperature=0.2,
            top_p=1.0,
            max_tokens=4096,
            timeout=120,
            extra_body={},
        )


# ---------------------------------------------------------------------------
# Provider registry and factory
# ---------------------------------------------------------------------------
_PROVIDERS: Dict[str, type[LLMProvider]] = {
    "nvidia": NVIDIAProvider,
    "openrouter": OpenRouterProvider,
}


def get_provider() -> LLMProvider:
    """Return the active provider instance based on LLM_PROVIDER env.

    Falls back to OpenRouter for backward compatibility when unset.
    Auto-detects NVIDIA if NVIDIA_API_KEY is set.
    """
    name = os.getenv("LLM_PROVIDER", "").strip().lower()

    if name and name in _PROVIDERS:
        provider = _PROVIDERS[name]()
        if provider.is_configured:
            return provider
        logger.warning(
            "LLM_PROVIDER=%s but its API key is not set; falling back.", name,
        )

    # Auto-detect: prefer NVIDIA if NVIDIA_API_KEY is set, else OpenRouter.
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    if not nvidia_key:
        try:
            import streamlit as st  # type: ignore
            nvidia_key = st.secrets.get("NVIDIA_API_KEY", "")
        except Exception:
            pass
    if nvidia_key:
        return NVIDIAProvider()

    return OpenRouterProvider()


def list_providers() -> List[Dict[str, Any]]:
    """Return info for all registered providers (no secrets)."""
    return [cls().info() for cls in _PROVIDERS.values()]


def get_provider_info() -> Dict[str, Any]:
    """Return info for the active provider (no secrets)."""
    return get_provider().info()


def provider_health_check() -> Dict[str, Any]:
    """Run a health check on the active provider (no secrets, no LLM call)."""
    return get_provider().health_check()
