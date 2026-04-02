"""Tests for SwivalCompleter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from prompt_toolkit.document import Document

from swival.completer import SwivalCompleter
from swival.skills import SkillInfo


@pytest.fixture
def catalog():
    return {
        "frontend-design": SkillInfo(
            name="frontend-design",
            description="Build frontend",
            path=Path("."),
            is_local=False,
        ),
        "security": SkillInfo(
            name="security",
            description="Security review",
            path=Path("."),
            is_local=False,
        ),
        "simplify": SkillInfo(
            name="simplify",
            description="Simplify code",
            path=Path("."),
            is_local=False,
        ),
    }


@pytest.fixture
def completer(catalog):
    return SwivalCompleter(skills_catalog=catalog)


def _completions(completer, text):
    """Return list of completion texts for the given input."""
    doc = Document(text, cursor_position=len(text))
    return [c.text for c in completer.get_completions(doc, None)]


# -- slash commands ---------------------------------------------------------


class TestSlashCommands:
    def test_prefix_match(self, completer):
        results = _completions(completer, "/com")
        assert "/compact" in results
        assert "/continue" not in results

    def test_prefix_con(self, completer):
        results = _completions(completer, "/con")
        assert "/continue" in results
        assert "/compact" not in results

    def test_exact_match(self, completer):
        results = _completions(completer, "/help")
        assert "/help" in results

    def test_no_match(self, completer):
        assert _completions(completer, "/xyz") == []

    def test_bare_slash_yields_all(self, completer):
        from swival.repl_commands import REPL_COMMANDS

        results = _completions(completer, "/")
        assert len(results) == len(REPL_COMMANDS)

    def test_case_insensitive(self, completer):
        results = _completions(completer, "/HEL")
        assert "/help" in results

    def test_display_meta_present(self, completer):
        doc = Document("/help", cursor_position=5)
        completions = list(completer.get_completions(doc, None))
        metas = [c.display_meta for c in completions if c.text == "/help"]
        assert metas and metas[0]


# -- command arguments ------------------------------------------------------


class TestCommandArguments:
    def test_add_dir_offers_directories(self, completer, tmp_path):
        subdir = tmp_path / "mydir"
        subdir.mkdir()
        results = _completions(completer, f"/add-dir {tmp_path}/")
        assert any("mydir" in r for r in results)

    def test_add_dir_ro_offers_directories(self, completer, tmp_path):
        subdir = tmp_path / "testdir"
        subdir.mkdir()
        results = _completions(completer, f"/add-dir-ro {tmp_path}/")
        assert any("testdir" in r for r in results)

    def test_simplify_no_path_completion(self, completer):
        assert _completions(completer, "/simplify ") == []

    def test_remember_no_completion(self, completer):
        assert _completions(completer, "/remember ") == []

    def test_status_no_completion(self, completer):
        assert _completions(completer, "/status ") == []

    def test_unknown_command_no_completion(self, completer):
        assert _completions(completer, "/unknown arg") == []


# -- custom commands --------------------------------------------------------


class TestCustomCommands:
    def test_prefix_match(self, completer):
        with patch(
            "swival.agent.discover_custom_commands",
            return_value=["deploy", "status"],
        ):
            results = _completions(completer, "!dep")
        assert "!deploy" in results
        assert "!status" not in results

    def test_bare_bang_lists_all(self, completer):
        with patch(
            "swival.agent.discover_custom_commands",
            return_value=["deploy", "status"],
        ):
            results = _completions(completer, "!")
        assert "!deploy" in results
        assert "!status" in results

    def test_no_commands_dir(self, completer):
        with patch("swival.agent.discover_custom_commands", return_value=[]):
            assert _completions(completer, "!foo") == []

    def test_no_completion_after_space(self, completer):
        """!cmd args should not trigger completion."""
        assert _completions(completer, "!deploy staging") == []

    def test_case_insensitive_on_windows(self, completer):
        """On Windows, runtime uses case-insensitive matching."""
        with (
            patch(
                "swival.agent.discover_custom_commands",
                return_value=["deploy"],
            ),
            patch("swival.completer.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            results = _completions(completer, "!DEP")
        assert "!deploy" in results

    def test_case_sensitive_on_unix(self, completer):
        with (
            patch(
                "swival.agent.discover_custom_commands",
                return_value=["deploy"],
            ),
            patch("swival.completer.sys") as mock_sys,
        ):
            mock_sys.platform = "linux"
            results = _completions(completer, "!DEP")
        assert results == []


# -- skill mentions ---------------------------------------------------------


class TestSkillMentions:
    def test_prefix_match(self, completer):
        results = _completions(completer, "$front")
        assert "$frontend-design" in results

    def test_mid_sentence(self, completer):
        results = _completions(completer, "use $fr")
        assert "$frontend-design" in results

    def test_no_boundary(self, completer):
        assert _completions(completer, "foo$bar") == []

    def test_empty_catalog(self):
        c = SwivalCompleter(skills_catalog={})
        assert _completions(c, "$front") == []

    def test_bare_dollar_yields_all(self, completer, catalog):
        results = _completions(completer, "$")
        assert len(results) == len(catalog)

    def test_display_meta_present(self, completer):
        doc = Document("$security", cursor_position=9)
        completions = list(completer.get_completions(doc, None))
        metas = [c.display_meta for c in completions if c.text == "$security"]
        assert metas and metas[0]


# -- plain text (no completion) ---------------------------------------------


class TestPlainText:
    def test_no_completion_for_prose(self, completer):
        assert _completions(completer, "tell me about") == []

    def test_no_completion_for_empty(self, completer):
        assert _completions(completer, "") == []
