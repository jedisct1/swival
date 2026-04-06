"""Tests for project-root auto-discovery."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

from swival.agent import _find_project_root


class TestFindProjectRoot:
    """Unit tests for _find_project_root."""

    def test_cwd_has_git(self, tmp_path):
        (tmp_path / ".git").mkdir()
        assert _find_project_root(tmp_path) == tmp_path

    def test_cwd_has_swival_toml(self, tmp_path):
        (tmp_path / "swival.toml").touch()
        assert _find_project_root(tmp_path) == tmp_path

    def test_subdirectory_of_git_project(self, tmp_path):
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        assert _find_project_root(sub) == tmp_path

    def test_subdirectory_of_swival_toml_project(self, tmp_path):
        (tmp_path / "swival.toml").touch()
        sub = tmp_path / "deep" / "nested"
        sub.mkdir(parents=True)
        assert _find_project_root(sub) == tmp_path

    def test_dot_swival_alone_does_not_trigger(self, tmp_path):
        sub = tmp_path / "child"
        sub.mkdir()
        (sub / ".swival").mkdir()
        assert _find_project_root(sub) == sub

    def test_no_markers_returns_start(self, tmp_path):
        sub = tmp_path / "x"
        sub.mkdir()
        assert _find_project_root(sub) == sub

    def test_git_file_worktree(self, tmp_path):
        (tmp_path / ".git").write_text("gitdir: /somewhere/else")
        sub = tmp_path / "src"
        sub.mkdir()
        assert _find_project_root(sub) == tmp_path

    def test_innermost_project_wins(self, tmp_path):
        (tmp_path / ".git").mkdir()
        inner = tmp_path / "sub"
        inner.mkdir()
        (inner / ".git").mkdir()
        child = inner / "pkg"
        child.mkdir()
        assert _find_project_root(child) == inner


class TestBaseDirNormalization:
    """Integration tests for --base-dir normalization in main()."""

    def test_explicit_base_dir_overrides_discovery(self, tmp_path):
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "child"
        sub.mkdir()

        from swival.agent import build_parser
        from swival.config import _UNSET

        parser = build_parser()
        args = parser.parse_args(["--base-dir", str(sub), "hello"])

        assert args.base_dir == str(sub)
        assert args.base_dir is not _UNSET

    def test_unset_base_dir_triggers_discovery(self, tmp_path):
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)

        from swival.agent import build_parser
        from swival.config import _UNSET

        parser = build_parser()
        args = parser.parse_args(["hello"])

        assert args.base_dir is _UNSET

        with mock.patch("swival.agent.Path") as mock_path_cls:
            real_path = Path

            def path_side_effect(*a, **kw):
                return real_path(*a, **kw)

            mock_path_cls.side_effect = path_side_effect
            mock_path_cls.cwd = mock.Mock(return_value=sub)

            if args.base_dir is _UNSET:
                args.base_dir = str(_find_project_root(sub))
            else:
                args.base_dir = str(real_path(args.base_dir).resolve())

        assert args.base_dir == str(tmp_path)

    def test_explicit_base_dir_dot_stays_literal(self, tmp_path):
        """Explicit --base-dir . resolves to cwd, not project root."""
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "nested"
        sub.mkdir()

        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(["--base-dir", ".", "hello"])

        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(sub)
            args.base_dir = str(Path(args.base_dir).resolve())
        finally:
            os.chdir(old_cwd)

        assert args.base_dir == str(sub)

    def test_reviewer_mode_ignores_discovery(self, tmp_path):
        """Reviewer mode uses positional arg as base_dir, not discovery."""
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "subdir"
        sub.mkdir()

        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(["--reviewer-mode", str(sub)])

        assert args.reviewer_mode is True
        assert args.question == str(sub)

    def test_init_config_project_uses_discovered_root(self, tmp_path):
        """--init-config --project from subdirectory writes to discovered root."""
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "deep"
        sub.mkdir()

        from swival.agent import build_parser
        from swival.config import _UNSET

        parser = build_parser()
        args = parser.parse_args(["--init-config", "--project"])

        assert args.base_dir is _UNSET

        args.base_dir = str(_find_project_root(sub))

        base_dir = Path(args.base_dir)
        assert base_dir == tmp_path


def _run_main_capture_base_dir(tmp_path, *, cwd, extra_argv=None):
    """Run main() in a subprocess, capturing the resolved args.base_dir.

    Patches _run_main to print args.base_dir and exit, so no LLM call is made.
    """
    argv_parts = extra_argv or ["hello"]
    script = textwrap.dedent("""\
        import sys, os
        os.chdir({cwd!r})
        sys.argv = ["swival"] + {argv!r}

        from swival import agent

        def fake_run_main(args, report, _write_report, parser):
            print(args.base_dir)
            raise SystemExit(0)

        agent._run_main = fake_run_main
        agent.main()
    """).format(cwd=str(cwd), argv=argv_parts)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "SWIVAL_SKIP_ONBOARDING": "1"},
    )
    return result


class TestMainLevelDiscovery:
    """Tests that exercise the actual main() control flow."""

    def test_discovery_from_subdirectory(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        (root / ".git").mkdir()
        sub = root / "src" / "pkg"
        sub.mkdir(parents=True)

        result = _run_main_capture_base_dir(tmp_path, cwd=sub)
        assert result.returncode == 0
        assert result.stdout.strip() == str(root)

    def test_explicit_base_dir_via_main(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        (root / ".git").mkdir()
        sub = root / "lib"
        sub.mkdir()

        result = _run_main_capture_base_dir(
            tmp_path, cwd=sub, extra_argv=["--base-dir", str(sub), "hello"]
        )
        assert result.returncode == 0
        assert result.stdout.strip() == str(sub)

    def test_explicit_base_dir_dot_via_main(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        (root / ".git").mkdir()
        sub = root / "nested"
        sub.mkdir()

        result = _run_main_capture_base_dir(
            tmp_path, cwd=sub, extra_argv=["--base-dir", ".", "hello"]
        )
        assert result.returncode == 0
        assert result.stdout.strip() == str(sub)

    def test_no_markers_falls_back_to_cwd(self, tmp_path):
        bare = tmp_path / "bare"
        bare.mkdir()

        result = _run_main_capture_base_dir(tmp_path, cwd=bare)
        assert result.returncode == 0
        assert result.stdout.strip() == str(bare)
