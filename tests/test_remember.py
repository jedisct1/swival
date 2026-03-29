"""Tests for /remember live context patching."""

from pathlib import Path

from swival.agent import (
    _patch_system_instructions,
    _repl_remember,
)


def _make_agents_md(tmp_path: Path, conventions: str = "- old fact\n") -> Path:
    p = tmp_path / "AGENTS.md"
    p.write_text(
        f"## Workflow\n\nstuff\n\n## Conventions\n\n{conventions}",
        encoding="utf-8",
    )
    return p


def _sys_msg(content: str) -> dict:
    return {"role": "system", "content": content}


# -- Positive: existing block gets updated --


def test_remember_updates_existing_block(tmp_path, monkeypatch):
    _make_agents_md(tmp_path, "- old fact\n")
    old_block = "<agent-instructions>\n- old fact\n</agent-instructions>"
    messages = [_sys_msg(f"You are helpful.\n\n{old_block}\n\nMore stuff.")]

    # Patch global_config_dir so load_instructions doesn't read user-level files
    monkeypatch.setattr(
        "swival.config.global_config_dir",
        lambda: tmp_path / "_no_config",
    )
    _repl_remember("new fact", str(tmp_path), messages)

    content = messages[0]["content"]
    assert "new fact" in content
    assert "old fact" in content
    # Surrounding content preserved
    assert "You are helpful." in content
    assert "More stuff." in content


# -- Negative: no block → no mutation --


def test_remember_no_block_no_mutation(tmp_path, monkeypatch):
    _make_agents_md(tmp_path)
    original = "Custom system prompt with no agent instructions."
    messages = [_sys_msg(original)]

    monkeypatch.setattr(
        "swival.config.global_config_dir",
        lambda: tmp_path / "_no_config",
    )
    _repl_remember("new fact", str(tmp_path), messages)

    # System message untouched
    assert messages[0]["content"] == original
    # But file was still written
    assert "new fact" in (tmp_path / "AGENTS.md").read_text()


# -- Negative: no system message at all --


def test_patch_no_system_message(tmp_path):
    messages = []
    _patch_system_instructions(messages, str(tmp_path))
    assert messages == []

    messages = [{"role": "user", "content": "hi"}]
    _patch_system_instructions(messages, str(tmp_path))
    assert messages[0]["content"] == "hi"


# -- Negative: command provider session --


def test_remember_command_provider_no_mutation(tmp_path, monkeypatch):
    _make_agents_md(tmp_path)
    # Command provider prompt has no <agent-instructions>
    original = (
        "You are a CLI tool. Respond with shell commands.\n\n"
        "Available external tools:\n\n- read_file\n- write_file"
    )
    messages = [_sys_msg(original)]

    monkeypatch.setattr(
        "swival.config.global_config_dir",
        lambda: tmp_path / "_no_config",
    )
    _repl_remember("new fact", str(tmp_path), messages)

    assert messages[0]["content"] == original
