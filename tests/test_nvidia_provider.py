"""Unit tests for the NVIDIA provider and provider abstraction.

All tests mock the OpenAI client — no live NVIDIA API key is required.
Run with:  pytest tests/test_nvidia_provider.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path so `config.*` and `services.*` resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.providers import (
    NVIDIAProvider,
    OpenRouterProvider,
    ProviderConfig,
    _redact,
    get_provider,
)


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------
class TestRedaction:
    def test_redact_replaces_secret(self):
        text = "my key is nvapi-abc123"
        result = _redact(text, "nvapi-abc123")
        assert "nvapi-abc123" not in result
        assert "[REDACTED]" in result

    def test_redact_handles_empty(self):
        assert _redact("") == ""
        assert _redact(None) is None

    def test_redact_api_key_pattern(self):
        text = "OPENROUTER_API_KEY=sk-or-v1-abc123def"
        result = _redact(text)
        assert "sk-or-v1-abc123def" not in result
        assert "[REDACTED]" in result

    def test_redact_password_pattern(self):
        text = "PASSWORD=supersecret"
        result = _redact(text)
        assert "supersecret" not in result
        assert "[REDACTED]" in result


# ---------------------------------------------------------------------------
# NVIDIA provider initialization
# ---------------------------------------------------------------------------
class TestNVIDIAProvider:
    def test_config_defaults(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=False):
            os.environ.pop("NVIDIA_BASE_URL", None)
            os.environ.pop("NVIDIA_MODEL", None)
            provider = NVIDIAProvider()
            cfg = provider.config
            assert cfg.name == "nvidia"
            assert cfg.base_url == "https://integrate.api.nvidia.com/v1"
            assert cfg.default_model == "nvidia/nemotron-3-ultra-550b-a55b"
            assert cfg.api_key == "test-key"

    def test_config_custom_url(self):
        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": "k",
            "NVIDIA_BASE_URL": "https://custom.example.com/v1",
            "NVIDIA_MODEL": "custom-model",
        }, clear=False):
            provider = NVIDIAProvider()
            cfg = provider.config
            assert cfg.base_url == "https://custom.example.com/v1"
            assert cfg.default_model == "custom-model"

    def test_is_configured_true(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "k"}, clear=False):
            assert NVIDIAProvider().is_configured is True

    def test_is_configured_false(self):
        with patch.dict(os.environ, {}, clear=True):
            # Also need to avoid Streamlit secrets
            with patch("builtins.__import__", side_effect=ImportError):
                provider = NVIDIAProvider()
                assert provider.is_configured is False

    def test_reasoning_enabled_in_extra_body(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "k"}, clear=False):
            cfg = NVIDIAProvider().config
            assert "chat_template_kwargs" in cfg.extra_body
            assert cfg.extra_body["chat_template_kwargs"]["enable_thinking"] is True

    def test_temperature_default(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "k"}, clear=False):
            cfg = NVIDIAProvider().config
            assert cfg.temperature == 0.2

    def test_top_p_default(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "k"}, clear=False):
            cfg = NVIDIAProvider().config
            assert cfg.top_p == 0.95


# ---------------------------------------------------------------------------
# OpenRouter provider (backward compat)
# ---------------------------------------------------------------------------
class TestOpenRouterProvider:
    def test_config_defaults(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}, clear=False):
            os.environ.pop("OPENROUTER_MODEL", None)
            os.environ.pop("OPENROUTER_MODEL_DEFAULT", None)
            provider = OpenRouterProvider()
            cfg = provider.config
            assert cfg.name == "openrouter"
            assert cfg.base_url == "https://openrouter.ai/api/v1"
            assert cfg.api_key == "or-key"


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------
class TestProviderFactory:
    def test_auto_detect_nvidia(self):
        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": "nvkey",
            "OPENROUTER_API_KEY": "",
        }, clear=False):
            provider = get_provider()
            assert provider.provider_name == "nvidia"

    def test_explicit_openrouter(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "orkey",
            "NVIDIA_API_KEY": "",
        }, clear=False):
            provider = get_provider()
            assert provider.provider_name == "openrouter"

    def test_explicit_nvidia(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "nvidia",
            "NVIDIA_API_KEY": "nvkey",
        }, clear=False):
            provider = get_provider()
            assert provider.provider_name == "nvidia"

    def test_fallback_when_requested_provider_key_missing(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "nvidia",
            "NVIDIA_API_KEY": "",
            "OPENROUTER_API_KEY": "orkey",
        }, clear=False):
            provider = get_provider()
            # Should fall back to OpenRouter since NVIDIA key is missing
            assert provider.provider_name == "openrouter"


# ---------------------------------------------------------------------------
# Generate (mocked)
# ---------------------------------------------------------------------------
class TestGenerate:
    def test_generate_success(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "k"}, clear=False):
            provider = NVIDIAProvider()
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "Hello from Nemotron"
            mock_resp.usage = None
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            provider._client = mock_client
            result = provider.generate("test prompt")
            assert result == "Hello from Nemotron"
            mock_client.chat.completions.create.assert_called_once()

    def test_generate_missing_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.__import__", side_effect=ImportError):
                provider = NVIDIAProvider()
                with pytest.raises(RuntimeError, match="not configured"):
                    provider.generate("test")

    def test_generate_retries_on_timeout(self):
        from openai import APITimeoutError
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "k"}, clear=False):
            provider = NVIDIAProvider()
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "recovered"
            mock_resp.usage = None
            # First call times out, second succeeds
            mock_client.chat.completions.create.side_effect = [
                APITimeoutError(request=MagicMock()),
                mock_resp,
            ]
            provider._client = mock_client
            with patch("time.sleep"):
                result = provider.generate("test", retries=2)
            assert result == "recovered"
            assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
class TestHealthCheck:
    def test_health_check_not_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.__import__", side_effect=ImportError):
                provider = NVIDIAProvider()
                result = provider.health_check()
                assert result["configured"] is False
                assert result["provider"] == "nvidia"
                assert "not configured" in result["error"]

    def test_health_check_reachable(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "k"}, clear=False):
            provider = NVIDIAProvider()
            mock_client = MagicMock()
            mock_client.models.list.return_value = []
            provider._client = mock_client
            result = provider.health_check()
            assert result["configured"] is True
            assert result["reachable"] is True
            assert result["model"] == "nvidia/nemotron-3-ultra-550b-a55b"

    def test_health_check_no_secret_in_result(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "secret-key-123"}, clear=False):
            provider = NVIDIAProvider()
            mock_client = MagicMock()
            mock_client.models.list.side_effect = Exception("auth error: secret-key-123")
            provider._client = mock_client
            result = provider.health_check()
            assert "secret-key-123" not in str(result)


# ---------------------------------------------------------------------------
# Info (no secrets)
# ---------------------------------------------------------------------------
class TestInfo:
    def test_info_no_api_key(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "my-secret-key"}, clear=False):
            provider = NVIDIAProvider()
            info = provider.info()
            assert "api_key" not in info
            assert info["provider"] == "nvidia"
            assert info["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
            assert "my-secret-key" not in str(info)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------
class TestContextManager:
    def test_is_excluded_env(self):
        from services.context_manager import is_excluded
        from pathlib import Path
        assert is_excluded(Path(".env")) is True
        assert is_excluded(Path(".git/config")) is True
        assert is_excluded(Path("node_modules/express/index.js")) is True
        assert is_excluded(Path("__pycache__/module.cpython-312.pyc")) is True

    def test_is_excluded_not_excluded(self):
        from services.context_manager import is_excluded
        from pathlib import Path
        assert is_excluded(Path("main.py")) is False
        assert is_excluded(Path("src/app.js")) is False
        assert is_excluded(Path("README.md")) is False

    def test_filter_files(self):
        from services.context_manager import filter_files
        files = ["main.py", ".env", "app/__pycache__/x.pyc", "README.md", "node_modules/x.js"]
        result = filter_files(files)
        assert ".env" not in result
        assert "node_modules/x.js" not in result
        assert "main.py" in result
        assert "README.md" in result

    def test_redact_secrets(self):
        from services.context_manager import redact_secrets
        text = "OPENROUTER_API_KEY=sk-or-v1-abc123"
        result = redact_secrets(text)
        assert "sk-or-v1-abc123" not in result
        assert "[REDACTED]" in result


# ---------------------------------------------------------------------------
# Command policy
# ---------------------------------------------------------------------------
class TestCommandPolicy:
    def test_pytest_is_safe(self):
        from services.command_policy import classify_command, CommandRisk
        from pathlib import Path
        d = classify_command("pytest", ".", Path("/tmp"))
        assert d.risk == CommandRisk.SAFE

    def test_rm_rf_is_blocked(self):
        from services.command_policy import classify_command, CommandRisk
        from pathlib import Path
        d = classify_command("rm -rf /", ".", Path("/tmp"))
        assert d.risk == CommandRisk.BLOCKED

    def test_pip_install_needs_approval(self):
        from services.command_policy import classify_command, CommandRisk
        from pathlib import Path
        d = classify_command("pip install requests", ".", Path("/tmp"))
        assert d.risk == CommandRisk.APPROVAL_REQUIRED

    def test_env_access_is_blocked(self):
        from services.command_policy import classify_command, CommandRisk
        from pathlib import Path
        d = classify_command("cat .env", ".", Path("/tmp"))
        assert d.risk == CommandRisk.BLOCKED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])