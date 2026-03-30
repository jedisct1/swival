"""Nono sandbox backend stub."""

import shutil

from .sandbox import SandboxBackend, SandboxError


class NonoBackend(SandboxBackend):
    name = "nono"

    def __init__(self, profile=None, mode=None, rollback=False):
        self.profile = profile
        self.mode = mode
        self.rollback = rollback

    def validate(self):
        if shutil.which("nono") is None:
            raise SandboxError(
                "nono is not installed or not on PATH. "
                "Install it from https://github.com/nichochar/nono"
            )

    def bootstrap(self, base_dir):
        raise NotImplementedError(
            "nono backend bootstrap is not yet implemented — "
            "waiting for stable nono API"
        )

    def metadata(self):
        return {
            "backend": "nono",
            "profile": self.profile,
            "mode": self.mode,
            "rollback": self.rollback,
        }
