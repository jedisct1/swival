"""Sandbox backend interface and host implementation."""

from typing import Any


class SandboxError(Exception):
    """Raised when a sandbox backend is unavailable or misconfigured."""


class SandboxBackend:
    name: str = "base"

    def validate(self) -> None:
        pass

    def bootstrap(self, base_dir: str) -> None:
        pass

    def metadata(self) -> dict[str, Any]:
        return {"backend": self.name}


class HostBackend(SandboxBackend):
    name = "host"


def create_backend(name: str, config: dict) -> SandboxBackend:
    if name == "host":
        return HostBackend()
    if name == "agentfs":
        from .sandbox_agentfs import AgentFSBackend

        return AgentFSBackend(
            session_id=config.get("sandbox_session"),
            strict_read=config.get("sandbox_strict_read", False),
            auto_session=config.get("sandbox_auto_session", True),
        )
    if name == "nono":
        from .sandbox_nono import NonoBackend

        return NonoBackend(
            profile=config.get("nono_profile"),
            mode=config.get("nono_mode"),
            rollback=config.get("nono_rollback", False),
        )
    raise SandboxError(f"unknown sandbox backend: {name!r}")
