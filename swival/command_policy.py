from __future__ import annotations

import os
import re
import sys
import tomllib

from rich.console import Console
from rich.text import Text

_WRAPPER_PATTERNS: list[tuple[list[str], int | None]] = [
    (["uv", "run"], 1),
    (["uv", "pip", "install"], None),
    (["python3", "-m"], 1),
    (["python", "-m"], 1),
    (["cargo", "test"], None),
    (["go", "test"], None),
    (["npm", "run"], 1),
    (["npm", "test"], None),
    (["npm", "install"], None),
    (["pip", "install"], None),
    (["git", "push"], None),
    (["git", "reset"], None),
    (["git", "clean"], None),
]

_INTERPRETERS = {"python3", "python", "bash", "sh", "node", "ruby", "perl"}

_TEMP_SCRIPT_RE = re.compile(r"(/tmp/swival-|\.swival/tmp/|/tmp/)")

HIGH_RISK_BUCKETS = {
    "rm",
    "git push",
    "git reset",
    "git clean",
    "docker",
    "kubectl",
    "curl",
    "wget",
    "npm install",
    "pip install",
    "uv pip install",
}


def normalize_bucket(argv: list[str]) -> str:
    if not argv:
        return ""

    for prefix, extra_count in _WRAPPER_PATTERNS:
        plen = len(prefix)
        if len(argv) >= plen and argv[:plen] == prefix:
            if extra_count is not None and len(argv) > plen:
                return " ".join(prefix + argv[plen : plen + extra_count])
            return " ".join(prefix)

    basename = os.path.basename(argv[0])
    if basename in _INTERPRETERS and len(argv) >= 2:
        if _TEMP_SCRIPT_RE.search(argv[1]):
            return f"{basename} <temp-script>"

    return basename


def is_high_risk(bucket: str) -> bool:
    return bucket in HIGH_RISK_BUCKETS


class CommandPolicy:
    def __init__(
        self,
        mode: str,
        allowed_basenames: set[str] | None = None,
        approved_buckets: set[str] | None = None,
    ):
        self.mode = mode
        self.allowed_basenames = allowed_basenames or set()
        self.approved_buckets = set(approved_buckets or ())
        self.denied_buckets: set[str] = set()
        self.always_ask_buckets: set[str] = set()

    def check(self, argv: list[str], is_subagent: bool = False) -> str | None:
        if self.mode == "full":
            return None
        if self.mode == "none":
            return "error: commands are disabled (commands=none). Adjust your plan."
        if self.mode == "allowlist":
            basename = os.path.basename(argv[0])
            if basename in self.allowed_basenames:
                return None
            return (
                f"error: command {basename!r} is not in the allowed list. "
                f"Allowed: {', '.join(sorted(self.allowed_basenames))}."
            )

        bucket = normalize_bucket(argv)

        if bucket in self.denied_buckets:
            return (
                f"error: user denied command bucket {bucket!r}. "
                f"Do not retry this command or any equivalent variant. Adjust your plan."
            )

        if bucket in self.approved_buckets and bucket not in self.always_ask_buckets:
            return None

        if is_subagent:
            return (
                f"error: command bucket {bucket!r} is not approved. "
                f"Subagents cannot prompt for approval. "
                f"Run this command from the main agent, or pre-approve the bucket in config."
            )

        return f"needs_approval:{bucket}"

    def approve_bucket(self, bucket: str) -> None:
        self.approved_buckets.add(bucket)
        self.denied_buckets.discard(bucket)

    def deny_bucket(self, bucket: str) -> None:
        self.denied_buckets.add(bucket)
        self.approved_buckets.discard(bucket)

    def mark_always_ask(self, bucket: str) -> None:
        self.always_ask_buckets.add(bucket)
        self.approved_buckets.discard(bucket)


def prompt_approval(bucket: str, high_risk: bool = False) -> str:
    console = Console(stderr=True)

    label = Text(bucket, style="bold")
    if high_risk:
        console.print(Text("⚠ high-risk ", style="bold red"), label, end="")
    else:
        console.print(Text("? ", style="bold yellow"), label, end="")

    if high_risk:
        hint = " [enter=deny / y=allow / p=persist / o=once / a=always-ask]: "
    else:
        hint = " [enter=allow / n=deny / p=persist / o=once / a=always-ask]: "

    sys.stderr.write(hint)
    sys.stderr.flush()

    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "deny"

    if answer == "":
        return "deny" if high_risk else "allow"
    if answer in ("y", "yes"):
        return "allow"
    if answer == "p":
        return "persist"
    if answer in ("n", "no"):
        return "deny"
    if answer == "o":
        return "once"
    if answer in ("a", "always"):
        return "always_ask"
    return "deny" if high_risk else "allow"


def persist_approved_bucket(bucket: str, base_dir: str) -> None:
    toml_path = os.path.join(base_dir, "swival.toml")

    existing = ""
    if os.path.isfile(toml_path):
        with open(toml_path, "r") as f:
            existing = f.read()

    try:
        data = tomllib.loads(existing)
    except Exception:
        data = {}

    current = data.get("approved_buckets", [])
    if bucket in current:
        return

    if "approved_buckets" in data:
        idx = existing.rfind("]", existing.index("approved_buckets"))
        before = existing[:idx].rstrip()
        after = existing[idx:]
        if before.endswith("["):
            new_content = before + f"\n    {bucket!r},\n" + after
        else:
            new_content = before + f",\n    {bucket!r},\n" + after
    else:
        if existing and not existing.endswith("\n"):
            existing += "\n"
        new_content = existing + f"\napproved_buckets = [\n    {bucket!r},\n]\n"

    with open(toml_path, "w") as f:
        f.write(new_content)
