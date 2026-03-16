"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_global_skills(monkeypatch):
    """Prevent all tests from picking up real ~/.agents/skills/ or ~/.config/swival/skills/.

    Global skill discovery scans Path.home() / ".agents" / "skills" and
    config.global_config_dir() / "skills".  Without isolation, tests that
    create a Session (or call discover_skills) on a machine with real global
    skills become environment-dependent.
    """
    monkeypatch.setattr("swival.skills._global_skill_dirs", lambda: [])
