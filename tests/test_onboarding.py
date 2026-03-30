"""Tests for swival.onboarding — first-run interactive setup wizard."""

import argparse
import types
from swival.onboarding import (
    run_onboarding,
    render_minimal_config,
    _mask_secret,
    _toml_escape,
)
from swival.agent import _should_try_onboarding
from swival.config import _UNSET


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Build a minimal argparse namespace for onboarding trigger tests."""
    defaults = {
        "provider": _UNSET,
        "profile": None,
        "reviewer_mode": False,
        "serve": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# _should_try_onboarding
# ---------------------------------------------------------------------------


class TestShouldTryOnboarding:
    def test_true_on_fresh_interactive_start(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args()
        assert _should_try_onboarding(args, tmp_path) is True

    def test_false_when_global_config_exists(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        (cfg_dir / "config.toml").write_text('provider = "lmstudio"\n')
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: cfg_dir)
        args = _make_args()
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_project_config_exists(self, tmp_path, monkeypatch):
        (tmp_path / "swival.toml").write_text('provider = "lmstudio"\n')
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args()
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_provider_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args(provider="lmstudio")
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_profile_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args(profile="fast")
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_stdin_piped(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: False))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args()
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_stderr_not_tty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: False))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args()
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_reviewer_mode(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args(reviewer_mode=True)
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_serve_mode(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: tmp_path / "cfg")
        args = _make_args(serve=True)
        assert _should_try_onboarding(args, tmp_path) is False

    def test_false_when_skip_marker_exists(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        (cfg_dir / ".onboarding-skipped").write_text("")
        monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("sys.stderr", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("swival.config.global_config_dir", lambda: cfg_dir)
        args = _make_args()
        assert _should_try_onboarding(args, tmp_path) is False


# ---------------------------------------------------------------------------
# render_minimal_config
# ---------------------------------------------------------------------------


class TestRenderMinimalConfig:
    def test_basic_provider_only(self):
        result = render_minimal_config({"provider": "lmstudio"})
        assert 'provider = "lmstudio"' in result
        assert result.startswith("# Swival config")

    def test_header_comment(self):
        result = render_minimal_config({"provider": "lmstudio"})
        assert "# Run `swival --init-config` to see all available options." in result

    def test_multiple_keys_ordered(self):
        settings = {
            "provider": "openrouter",
            "model": "openai/gpt-4.1",
            "api_key": "sk-123",
            "max_context_tokens": 131072,
        }
        result = render_minimal_config(settings)
        config_lines = [
            ln for ln in result.split("\n") if ln and not ln.startswith("#")
        ]
        assert config_lines[0] == 'provider = "openrouter"'
        assert config_lines[1] == 'model = "openai/gpt-4.1"'
        assert config_lines[2] == 'api_key = "sk-123"'
        assert config_lines[3] == "max_context_tokens = 131072"

    def test_integer_values(self):
        result = render_minimal_config(
            {"provider": "generic", "max_context_tokens": 65536}
        )
        assert "max_context_tokens = 65536" in result

    def test_omits_unknown_keys(self):
        result = render_minimal_config({"provider": "lmstudio", "unknown_key": "val"})
        assert "unknown_key" not in result

    def test_toml_escaping(self):
        result = render_minimal_config({"provider": "generic", "model": 'a"b\\c'})
        assert r'model = "a\"b\\c"' in result

    def test_trailing_newline(self):
        result = render_minimal_config({"provider": "lmstudio"})
        assert result.endswith("\n")


# ---------------------------------------------------------------------------
# _mask_secret / _toml_escape
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_mask_secret_long(self):
        assert _mask_secret("sk-abcdef1234") == "*********1234"

    def test_mask_secret_short(self):
        assert _mask_secret("abc") == "****"

    def test_toml_escape_backslash(self):
        assert _toml_escape("a\\b") == "a\\\\b"

    def test_toml_escape_quote(self):
        assert _toml_escape('a"b') == 'a\\"b'

    def test_toml_escape_newline(self):
        assert _toml_escape("a\nb") == "a\\nb"


# ---------------------------------------------------------------------------
# run_onboarding integration tests
# ---------------------------------------------------------------------------


class TestRunOnboarding:
    def _patch_prompts(self, monkeypatch, responses):
        """Patch _session.prompt to return values from a list in order."""
        call_idx = {"i": 0}

        def fake_prompt(*args, **kwargs):
            if call_idx["i"] < len(responses):
                val = responses[call_idx["i"]]
                call_idx["i"] += 1
                if val is KeyboardInterrupt:
                    raise KeyboardInterrupt
                return val
            return ""

        monkeypatch.setattr("swival.onboarding._session.prompt", fake_prompt)

    def test_cancel_at_welcome(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        self._patch_prompts(
            monkeypatch,
            [
                "2",  # Cancel at welcome
            ],
        )
        result = run_onboarding()
        assert result is None
        assert not (cfg_dir / "config.toml").exists()
        assert not (cfg_dir / ".onboarding-skipped").exists()

    def test_ctrl_c_exits_cleanly(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        self._patch_prompts(
            monkeypatch,
            [
                "1",  # Continue at welcome
                KeyboardInterrupt,
            ],
        )
        result = run_onboarding()
        assert result is None
        assert not (cfg_dir / "config.toml").exists()
        assert not (cfg_dir / ".onboarding-skipped").exists()

    def test_successful_lmstudio_onboarding(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        self._patch_prompts(
            monkeypatch,
            [
                "1",  # Continue at welcome
                "1",  # LM Studio provider
                "y",  # Use default server
                "",  # Model blank (auto-discovery)
                "1",  # Yes, write config
            ],
        )
        result = run_onboarding()
        assert result is not None
        assert result.exists()
        content = result.read_text()
        assert 'provider = "lmstudio"' in content
        assert "model" not in content

    def test_successful_openrouter_with_env_var(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        self._patch_prompts(
            monkeypatch,
            [
                "1",  # Continue at welcome
                "3",  # OpenRouter
                "openai/gpt-4.1",  # model
                "1",  # I'll set OPENROUTER_API_KEY myself
                "",  # skip context window
                "1",  # Yes, write config
            ],
        )
        result = run_onboarding()
        assert result is not None
        content = result.read_text()
        assert 'provider = "openrouter"' in content
        assert 'model = "openai/gpt-4.1"' in content
        assert "api_key" not in content

    def test_successful_generic_with_api_key(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        self._patch_prompts(
            monkeypatch,
            [
                "1",  # Continue at welcome
                "5",  # Generic OpenAI-compatible
                "http://localhost:11434",  # base URL
                "qwen3:32b",  # model
                "2",  # Enter API key now
                "sk-test-key",  # the key
                "",  # skip context window
                "1",  # Yes, write config
            ],
        )
        result = run_onboarding()
        assert result is not None
        content = result.read_text()
        assert 'provider = "generic"' in content
        assert 'base_url = "http://localhost:11434"' in content
        assert 'model = "qwen3:32b"' in content
        assert 'api_key = "sk-test-key"' in content

    def test_cancel_at_confirmation_writes_skip_marker(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        self._patch_prompts(
            monkeypatch,
            [
                "1",  # Continue at welcome
                "1",  # LM Studio
                "y",  # Default server
                "",  # No model
                "3",  # Cancel at confirmation
            ],
        )
        result = run_onboarding()
        assert result is None
        assert not (cfg_dir / "config.toml").exists()
        assert (cfg_dir / ".onboarding-skipped").exists()

    def test_start_over_loops_back(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        self._patch_prompts(
            monkeypatch,
            [
                "1",  # Continue at welcome
                "1",  # LM Studio (first attempt)
                "y",  # Default server
                "",  # No model
                "2",  # Start over
                "2",  # ChatGPT (second attempt)
                "gpt-4.1",  # model
                "",  # skip reasoning effort
                "1",  # Yes, write config
            ],
        )
        result = run_onboarding()
        assert result is not None
        content = result.read_text()
        assert 'provider = "chatgpt"' in content
        assert 'model = "gpt-4.1"' in content

    def test_no_overwrite_existing_config(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir(parents=True)
        existing = cfg_dir / "config.toml"
        existing.write_text('provider = "lmstudio"\n')
        monkeypatch.setattr("swival.onboarding.global_config_dir", lambda: cfg_dir)
        self._patch_prompts(
            monkeypatch,
            [
                "1",  # Continue at welcome
                "2",  # ChatGPT
                "gpt-4.1",
                "",
                "1",  # Yes, write config
            ],
        )
        result = run_onboarding()
        assert result is None
        assert existing.read_text() == 'provider = "lmstudio"\n'
