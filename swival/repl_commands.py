"""Single registry of REPL slash commands."""

from typing import NamedTuple


class CommandInfo(NamedTuple):
    """Metadata for a REPL slash command."""

    desc: str
    arg: str | None = None
    arg_type: str | None = None


REPL_COMMANDS: dict[str, CommandInfo] = {
    "/add-dir": CommandInfo(
        desc="Grant read+write access to a directory",
        arg="<path>",
        arg_type="dir_path",
    ),
    "/add-dir-ro": CommandInfo(
        desc="Grant read-only access to a directory",
        arg="<path>",
        arg_type="dir_path",
    ),
    "/clear": CommandInfo(desc="Reset conversation to initial state"),
    "/compact": CommandInfo(
        desc="Compress context (--drop removes middle turns)",
        arg="[--drop]",
    ),
    "/continue": CommandInfo(desc="Reset turn counter and continue the agent loop"),
    "/copy": CommandInfo(desc="Copy last output to clipboard"),
    "/exit": CommandInfo(desc="Exit the REPL"),
    "/extend": CommandInfo(desc="Double max turns, or set to N", arg="[N]"),
    "/help": CommandInfo(desc="Show this help message"),
    "/init": CommandInfo(
        desc="Scan project for build/test/lint workflow and conventions, write AGENTS.md",
    ),
    "/learn": CommandInfo(desc="Review session for mistakes and persist to memory"),
    "/new": CommandInfo(desc="Reset conversation to initial state"),
    "/profile": CommandInfo(
        desc="Switch LLM profile (no arg = list, - = revert)",
        arg="[name]",
    ),
    "/quit": CommandInfo(desc="Exit the REPL"),
    "/remember": CommandInfo(
        desc="Add a durable project fact to AGENTS.md",
        arg="<text>",
    ),
    "/restore": CommandInfo(desc="Summarize & collapse since checkpoint"),
    "/save": CommandInfo(desc="Set a context checkpoint", arg="[label]"),
    "/simplify": CommandInfo(
        desc="Simplify codebase (optionally scoped to focus area)",
        arg="[focus]",
    ),
    "/status": CommandInfo(desc="Show session stats (model, context, turns, state)"),
    "/tools": CommandInfo(desc="List all available tools"),
    "/unsave": CommandInfo(desc="Cancel active checkpoint"),
}
