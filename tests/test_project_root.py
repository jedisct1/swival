"""Tests for project-root auto-discovery."""

import os
import subprocess
import sys
import textwrap

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
    """Argparse-level tests for --base-dir default."""

    def test_explicit_base_dir_is_not_unset(self, tmp_path):
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "child"
        sub.mkdir()

        from swival.agent import build_parser
        from swival.config import _UNSET

        parser = build_parser()
        args = parser.parse_args(["--base-dir", str(sub), "hello"])

        assert args.base_dir == str(sub)
        assert args.base_dir is not _UNSET

    def test_reviewer_mode_uses_positional_arg(self, tmp_path):
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "subdir"
        sub.mkdir()

        from swival.agent import build_parser

        parser = build_parser()
        args = parser.parse_args(["--reviewer-mode", str(sub)])

        assert args.reviewer_mode is True
        assert args.question == str(sub)


def _run_main_capture_base_dir(tmp_path, *, cwd, extra_argv=None):
    """Patches _run_main to print args.base_dir and exit (no LLM call)."""
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
