"""Tests for the command middleware feature."""

import json
import stat
import sys

import pytest

from swival.command_middleware import run_command_middleware
from swival.tools import NormalizedCommandCall, dispatch


def _make_middleware_script(
    tmp_path, response: dict | None, exit_code: int = 0, bad_json: bool = False
) -> str:
    """Write a tiny middleware script that emits a fixed response and return its path."""
    script = tmp_path / "middleware.py"
    if bad_json:
        body = 'print("not json")'
    elif response is None:
        body = f"import sys; sys.exit({exit_code})"
    else:
        body = f"print({json.dumps(json.dumps(response))})"
    script.write_text(f"#!/usr/bin/env python3\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return f"{sys.executable} {script}"


def _normalized(mode="argv", command=None):
    if command is None:
        command = ["echo", "hi"] if mode == "argv" else "echo hi"
    return NormalizedCommandCall(mode=mode, command=command)


class TestRunCommandMiddleware:
    def test_passthrough(self, tmp_path):
        cmd = _make_middleware_script(tmp_path, {"action": "allow"})
        result = run_command_middleware(
            cmd,
            tool_name="run_command",
            normalized=_normalized(),
            base_dir=str(tmp_path),
            timeout=30,
            is_subagent=False,
        )
        assert result.action == "allow"
        assert result.normalized is None
        assert result.warning is None

    def test_shell_rewrite(self, tmp_path):
        cmd = _make_middleware_script(
            tmp_path, {"action": "allow", "mode": "shell", "command": "rtk git status"}
        )
        result = run_command_middleware(
            cmd,
            tool_name="run_shell_command",
            normalized=_normalized("shell", "git status"),
            base_dir=str(tmp_path),
            timeout=30,
            is_subagent=False,
        )
        assert result.action == "allow"
        assert result.normalized is not None
        assert result.normalized.mode == "shell"
        assert result.normalized.command == "rtk git status"

    def test_argv_rewrite(self, tmp_path):
        cmd = _make_middleware_script(
            tmp_path,
            {"action": "allow", "mode": "argv", "command": ["rtk", "git", "status"]},
        )
        result = run_command_middleware(
            cmd,
            tool_name="run_command",
            normalized=_normalized("argv", ["git", "status"]),
            base_dir=str(tmp_path),
            timeout=30,
            is_subagent=False,
        )
        assert result.action == "allow"
        assert result.normalized is not None
        assert result.normalized.mode == "argv"
        assert result.normalized.command == ["rtk", "git", "status"]

    def test_deny(self, tmp_path):
        cmd = _make_middleware_script(
            tmp_path, {"action": "deny", "reason": "not allowed"}
        )
        result = run_command_middleware(
            cmd,
            tool_name="run_command",
            normalized=_normalized(),
            base_dir=str(tmp_path),
            timeout=30,
            is_subagent=False,
        )
        assert result.action == "deny"
        assert result.reason == "not allowed"

    def test_nonzero_exit_fails_open(self, tmp_path):
        cmd = _make_middleware_script(tmp_path, None, exit_code=1)
        result = run_command_middleware(
            cmd,
            tool_name="run_command",
            normalized=_normalized(),
            base_dir=str(tmp_path),
            timeout=30,
            is_subagent=False,
        )
        assert result.action == "allow"
        assert result.warning is not None

    def test_bad_json_fails_open(self, tmp_path):
        cmd = _make_middleware_script(tmp_path, None, bad_json=True)
        result = run_command_middleware(
            cmd,
            tool_name="run_command",
            normalized=_normalized(),
            base_dir=str(tmp_path),
            timeout=30,
            is_subagent=False,
        )
        assert result.action == "allow"
        assert result.warning is not None

    def test_missing_executable_fails_open(self, tmp_path):
        result = run_command_middleware(
            "/nonexistent/binary",
            tool_name="run_command",
            normalized=_normalized(),
            base_dir=str(tmp_path),
            timeout=30,
            is_subagent=False,
        )
        assert result.action == "allow"
        assert result.warning is not None

    def test_unknown_action_fails_open(self, tmp_path):
        cmd = _make_middleware_script(tmp_path, {"action": "unknown"})
        result = run_command_middleware(
            cmd,
            tool_name="run_command",
            normalized=_normalized(),
            base_dir=str(tmp_path),
            timeout=30,
            is_subagent=False,
        )
        assert result.action == "allow"
        assert result.warning is not None

    def test_argv_mode_requires_list(self, tmp_path):
        cmd = _make_middleware_script(
            tmp_path, {"action": "allow", "mode": "argv", "command": "not-a-list"}
        )
        result = run_command_middleware(
            cmd,
            tool_name="run_command",
            normalized=_normalized(),
            base_dir=str(tmp_path),
            timeout=30,
            is_subagent=False,
        )
        assert result.action == "allow"
        assert result.warning is not None

    def test_shell_mode_requires_string(self, tmp_path):
        cmd = _make_middleware_script(
            tmp_path,
            {"action": "allow", "mode": "shell", "command": ["not", "a", "string"]},
        )
        result = run_command_middleware(
            cmd,
            tool_name="run_command",
            normalized=_normalized(),
            base_dir=str(tmp_path),
            timeout=30,
            is_subagent=False,
        )
        assert result.action == "allow"
        assert result.warning is not None

    def test_payload_contains_correct_fields(self, tmp_path):
        received_path = tmp_path / "payload.json"
        script = tmp_path / "capture.py"
        script.write_text(
            f"#!/usr/bin/env python3\nimport sys, json\n"
            f"data = json.load(sys.stdin)\n"
            f"open({str(received_path)!r}, 'w').write(json.dumps(data))\n"
            f'print(\'{{"action": "allow"}}\')\n'
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        cmd = f"{sys.executable} {script}"
        run_command_middleware(
            cmd,
            tool_name="run_command",
            normalized=NormalizedCommandCall(mode="argv", command=["git", "status"]),
            base_dir=str(tmp_path),
            timeout=30,
            is_subagent=True,
        )
        payload = json.loads(received_path.read_text())
        assert payload["phase"] == "before"
        assert payload["tool"] == "run_command"
        assert payload["mode"] == "argv"
        assert payload["command"] == ["git", "status"]
        assert payload["cwd"] == str(tmp_path)
        assert payload["is_subagent"] is True


class TestDispatchMiddleware:
    """Integration tests: middleware wired through dispatch()."""

    def _simple_cmd(self, tmp_path):
        """Return a command that echoes 'hello'."""
        if sys.platform == "win32":
            return ["cmd", "/c", "echo", "hello"]
        return ["echo", "hello"]

    def test_passthrough_executes_original(self, tmp_path):
        mw = _make_middleware_script(tmp_path, {"action": "allow"})
        result = dispatch(
            "run_command",
            {"command": ["echo", "hello"]},
            str(tmp_path),
            commands_unrestricted=True,
            shell_allowed=False,
            command_middleware=mw,
        )
        assert "hello" in result

    def test_rewrite_executes_rewritten_command(self, tmp_path):
        if sys.platform == "win32":
            pytest.skip("Shell rewrite test is Unix-only")
        mw = _make_middleware_script(
            tmp_path, {"action": "allow", "mode": "shell", "command": "echo rewritten"}
        )
        result = dispatch(
            "run_shell_command",
            {"command": "echo original"},
            str(tmp_path),
            commands_unrestricted=True,
            shell_allowed=True,
            command_middleware=mw,
        )
        assert "rewritten" in result

    def test_deny_returns_error(self, tmp_path):
        mw = _make_middleware_script(
            tmp_path, {"action": "deny", "reason": "blocked by policy"}
        )
        result = dispatch(
            "run_command",
            {"command": ["echo", "hello"]},
            str(tmp_path),
            commands_unrestricted=True,
            shell_allowed=False,
            command_middleware=mw,
        )
        assert result.startswith("error:")
        assert "blocked by middleware" in result
        assert "blocked by policy" in result

    def test_no_middleware_unchanged(self, tmp_path):
        result = dispatch(
            "run_command",
            {"command": ["echo", "hello"]},
            str(tmp_path),
            commands_unrestricted=True,
            shell_allowed=False,
        )
        assert "hello" in result

    def test_rewritten_command_obeys_policy(self, tmp_path):
        """Middleware cannot rewrite a command to bypass an allowlist."""
        mw = _make_middleware_script(
            tmp_path, {"action": "allow", "mode": "argv", "command": ["rm", "-rf", "/"]}
        )
        from swival.command_policy import CommandPolicy
        from swival.agent import resolve_commands

        allowed = resolve_commands(["echo"], str(tmp_path))
        policy = CommandPolicy("allowlist", allowed_basenames=set(allowed))
        result = dispatch(
            "run_command",
            {"command": ["echo", "hi"]},
            str(tmp_path),
            commands_unrestricted=False,
            shell_allowed=False,
            command_middleware=mw,
            command_policy=policy,
            resolved_commands=allowed,
        )
        assert result.startswith("error:")


class TestConfigPlumbing:
    """Verify that command_middleware has a proper None default so _UNSET never reaches validation."""

    def test_argparse_defaults_has_none(self):
        from swival.config import _ARGPARSE_DEFAULTS

        assert "command_middleware" in _ARGPARSE_DEFAULTS
        assert _ARGPARSE_DEFAULTS["command_middleware"] is None

    def test_apply_config_to_args_sweeps_unset_to_none(self):
        import argparse
        from swival.config import _UNSET, apply_config_to_args

        args = argparse.Namespace(command_middleware=_UNSET)
        apply_config_to_args(args, {})
        assert args.command_middleware is None

    def test_apply_config_to_args_threads_value(self):
        import argparse
        from swival.config import _UNSET, apply_config_to_args

        args = argparse.Namespace(command_middleware=_UNSET)
        apply_config_to_args(args, {"command_middleware": "./my-mw.py"})
        assert args.command_middleware == "./my-mw.py"
