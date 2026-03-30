"""Interactive first-run onboarding wizard.

Guides the user through provider selection and config creation on first run.
All output goes to stderr via Rich. Never writes to stdout.
"""

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.text import Text

from .config import global_config_dir

_console = Console(stderr=True)
_session = PromptSession()

# Provider definitions: (internal_name, display_label, description)
_PROVIDERS = [
    ("lmstudio", "LM Studio", "Local models on your machine"),
    ("chatgpt", "ChatGPT", "Use your ChatGPT Plus or Pro subscription"),
    ("openrouter", "OpenRouter", "Hosted models with one API key"),
    ("google", "Google Gemini", "Gemini through Google's API"),
    ("generic", "OpenAI-compatible", "A local or remote server you already run"),
    ("huggingface", "HuggingFace", "Hosted inference API"),
    ("bedrock", "AWS Bedrock", "Models through AWS"),
    ("command", "Command", "Use an external program as the backend (advanced)"),
]

# Ordered keys for minimal config output
_CONFIG_KEY_ORDER = [
    "provider",
    "model",
    "base_url",
    "api_key",
    "aws_profile",
    "max_context_tokens",
    "max_output_tokens",
    "reasoning_effort",
    "temperature",
    "top_p",
    "seed",
]

_SKIP_MARKER = ".onboarding-skipped"


def _skip_marker_path() -> Path:
    return global_config_dir() / _SKIP_MARKER


def _global_config_path() -> Path:
    return global_config_dir() / "config.toml"


def run_onboarding() -> Path | None:
    """Run the interactive onboarding wizard.

    Returns the path to the created config file, or None if canceled.
    """
    try:
        return _onboarding_flow()
    except (KeyboardInterrupt, EOFError):
        _console.print()
        _console.print(
            Text("No worries! Run swival again whenever you're ready.", style="dim")
        )
        return None


def _onboarding_flow() -> Path | None:
    """The main onboarding flow. Raises KeyboardInterrupt/EOFError on Ctrl-C."""

    # Step 1: Welcome
    _console.print()
    _console.print(Text("Hey there! Welcome to Swival.", style="bold"))
    _console.print()
    _console.print(
        "Swival is a coding agent that lives in your terminal. It can dig through\n"
        "your codebase, edit files, run commands, and pair with you in a live REPL."
    )
    _console.print()
    _console.print(
        "Looks like this is your first time here, so let's get you set up!\n"
        "I'll create a global config so Swival works in every project on your machine."
    )
    _console.print()

    choice = _prompt_choice("Sound good?", ["Let's go!", "Not right now"])
    if choice == 1:
        return None

    # Steps 2-4 loop (Start over returns here)
    while True:
        settings = _collect_settings()
        if settings is None:
            return None

        result = _preview_and_confirm(settings)
        if result == "yes":
            return _write_config(settings)
        elif result == "start_over":
            continue
        else:
            _write_skip_marker()
            return None


def _collect_settings() -> dict | None:
    """Steps 2-3: provider selection + provider-specific questions.

    Returns a settings dict or None if canceled.
    """
    _console.print()
    _console.print(Text("Pick your LLM provider:", style="bold"))
    _console.print()

    labels = []
    for _, display, desc in _PROVIDERS:
        labels.append(f"{display:<20s}{desc}")

    idx = _prompt_choice("Provider", labels)
    provider_name, provider_display, _ = _PROVIDERS[idx]

    settings = {"provider": provider_name}

    _console.print()
    _console.print(Text(f"Nice! Let's configure {provider_display}.", style="bold"))
    _console.print()

    if provider_name == "lmstudio":
        _ask_lmstudio(settings)
    elif provider_name == "chatgpt":
        _ask_chatgpt(settings)
    elif provider_name == "openrouter":
        _ask_openrouter(settings)
    elif provider_name == "google":
        _ask_google(settings)
    elif provider_name == "generic":
        _ask_generic(settings)
    elif provider_name == "huggingface":
        _ask_huggingface(settings)
    elif provider_name == "bedrock":
        _ask_bedrock(settings)
    elif provider_name == "command":
        _ask_command(settings)

    return settings


def _preview_and_confirm(settings: dict) -> str:
    """Step 4: Show preview and ask for confirmation.

    Returns "yes", "start_over", or "cancel".
    """
    _console.print()
    _console.print(Text("Here's what I'll write:", style="bold"))
    _console.print()

    dest = _global_config_path()
    _console.print(f"  Location: {dest}")
    for key in _CONFIG_KEY_ORDER:
        if key not in settings:
            continue
        val = settings[key]
        display_key = key.replace("_", " ").title()
        if key == "api_key":
            val = _mask_secret(val)
        _console.print(f"  {display_key}: {val}")
    _console.print()

    idx = _prompt_choice(
        "Write this config?", ["Looks good, write it!", "Start over", "Cancel"]
    )
    if idx == 0:
        return "yes"
    elif idx == 1:
        return "start_over"
    else:
        return "cancel"


def _write_config(settings: dict) -> Path | None:
    """Step 5: Write the config file and show success screen."""
    dest = _global_config_path()

    if dest.exists():
        _console.print()
        _console.print(
            Text(
                f"A config file already exists at {dest}. Not overwriting.",
                style="yellow",
            )
        )
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_minimal_config(settings))

    _console.print()
    _console.print(Text("You're all set!", style="bold green"))
    _console.print()
    _console.print(f"Config saved to:\n  {dest}")
    _console.print()
    _console.print("Give it a spin:")
    _console.print('  swival "summarize this repository"')
    _console.print("  swival")
    _console.print()
    _console.print(
        Text(
            "Tip: run swival with no arguments to open the REPL. Try --self-review\n"
            "for an extra quality pass, or --yolo when you want full autonomy.",
            style="dim",
        )
    )
    _console.print()
    _console.print("Happy building!")
    _console.print()

    return dest


def _write_skip_marker() -> None:
    """Write the global skip marker so onboarding doesn't re-prompt."""
    marker = _skip_marker_path()
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Provider-specific question flows
# ---------------------------------------------------------------------------


def _ask_lmstudio(s: dict) -> None:
    _console.print(
        Text(
            "Great pick! LM Studio is the fastest way to get going locally.\n"
            "Leave the model blank and Swival will auto-detect whatever you\n"
            "have loaded.",
            style="dim",
        )
    )
    _console.print()

    use_default = _prompt_confirm(
        "Use the default server at http://127.0.0.1:1234?", default=True
    )
    if not use_default:
        url = _prompt_text("Server URL", default="http://127.0.0.1:1234")
        if url and url != "http://127.0.0.1:1234":
            s["base_url"] = url

    model = _prompt_text("Model name (blank for auto-discovery)", default="")
    if model:
        s["model"] = model


def _ask_chatgpt(s: dict) -> None:
    _console.print(
        Text(
            "On first use Swival will pop open a quick device login in your browser.\n"
            "After that it remembers you automatically.",
            style="dim",
        )
    )
    _console.print()

    model = _prompt_text("Model name", default="gpt-5.4")
    if model:
        s["model"] = model

    effort = _prompt_text(
        "Reasoning effort (none/low/medium/high, blank to skip)", default=""
    )
    if effort:
        s["reasoning_effort"] = effort


def _ask_openrouter(s: dict) -> None:
    model = _prompt_text("Model (e.g. openai/gpt-5.4)", default="")
    if model:
        s["model"] = model

    _ask_api_key(s, env_var="OPENROUTER_API_KEY")

    ctx = _prompt_int("Max context tokens (blank to skip)", default=None)
    if ctx is not None:
        s["max_context_tokens"] = ctx


def _ask_google(s: dict) -> None:
    model = _prompt_text("Model (e.g. gemini-2.5-flash)", default="")
    if model:
        s["model"] = model

    _ask_api_key(s, env_var="GEMINI_API_KEY")


def _ask_generic(s: dict) -> None:
    url = _prompt_text("Base URL (e.g. http://127.0.0.1:11434)", default="")
    if url:
        s["base_url"] = url

    model = _prompt_text("Model name", default="")
    if model:
        s["model"] = model

    _ask_api_key(s, env_var="OPENAI_API_KEY")

    ctx = _prompt_int("Max context tokens (blank to skip)", default=None)
    if ctx is not None:
        s["max_context_tokens"] = ctx


def _ask_huggingface(s: dict) -> None:
    model = _prompt_text("Model (org/model, e.g. zai-org/GLM-5)", default="")
    if model:
        s["model"] = model

    _ask_api_key(s, env_var="HF_TOKEN", label="HuggingFace token")

    url = _prompt_text("Endpoint URL override (blank to skip)", default="")
    if url:
        s["base_url"] = url


def _ask_bedrock(s: dict) -> None:
    model = _prompt_text("Model (e.g. global.anthropic.claude-opus-4-6-v1)", default="")
    if model:
        s["model"] = model

    region = _prompt_text("AWS region (blank for default)", default="")
    if region:
        s["base_url"] = region

    profile = _prompt_text("AWS profile name (blank for default)", default="")
    if profile:
        s["aws_profile"] = profile


def _ask_command(s: dict) -> None:
    _console.print(
        Text(
            "This shells out to an external program as the LLM backend.\n"
            "The model value is the command Swival will run.",
            style="dim",
        )
    )
    _console.print()

    cmd = _prompt_text("Command to run as the backend", default="")
    if cmd:
        s["model"] = cmd


def _ask_api_key(s: dict, *, env_var: str, label: str = "API key") -> None:
    """Ask whether to store an API key in config or use an env var."""
    idx = _prompt_choice(
        label,
        [f"I'll set {env_var} myself", "Enter it now (stored in config)"],
    )
    if idx == 1:
        key = _prompt_text(label, default="", secret=True)
        if key:
            s["api_key"] = key


# ---------------------------------------------------------------------------
# Config rendering
# ---------------------------------------------------------------------------


def render_minimal_config(settings: dict) -> str:
    """Render a minimal TOML config string from onboarding settings."""
    lines = [
        "# Swival config, created by first-run setup.",
        "# Run `swival --init-config` to see all available options.",
        "",
    ]
    for key in _CONFIG_KEY_ORDER:
        if key not in settings:
            continue
        val = settings[key]
        if isinstance(val, bool):
            lines.append(f"{key} = {'true' if val else 'false'}")
        elif isinstance(val, int):
            lines.append(f"{key} = {val}")
        else:
            lines.append(f'{key} = "{_toml_escape(val)}"')
    lines.append("")  # trailing newline
    return "\n".join(lines)


def _toml_escape(s: str) -> str:
    """Escape a string for TOML double-quoted values."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _mask_secret(val: str) -> str:
    """Mask all but the last 4 characters of a secret."""
    if len(val) <= 4:
        return "****"
    return "*" * (len(val) - 4) + val[-4:]


# ---------------------------------------------------------------------------
# Prompt helpers (prompt_toolkit-based, output to stderr)
# ---------------------------------------------------------------------------


def _prompt_choice(label: str, choices: list[str]) -> int:
    """Present a numbered list and return the 0-based index of the selection."""
    for i, c in enumerate(choices, 1):
        _console.print(f"  {i}. {c}")
    _console.print()

    while True:
        raw = _session.prompt(
            HTML(f"<b>{label}</b> [1-{len(choices)}]: "),
        ).strip()
        try:
            n = int(raw)
            if 1 <= n <= len(choices):
                return n - 1
        except ValueError:
            pass
        _console.print(f"  Please enter a number between 1 and {len(choices)}.")


def _prompt_confirm(label: str, *, default: bool = True) -> bool:
    """Yes/no confirmation prompt."""
    hint = "Y/n" if default else "y/N"
    raw = (
        _session.prompt(
            HTML(f"<b>{label}</b> [{hint}]: "),
        )
        .strip()
        .lower()
    )
    if not raw:
        return default
    return raw in ("y", "yes")


def _prompt_text(label: str, *, default: str = "", secret: bool = False) -> str:
    """Free-text prompt with optional default."""
    if default:
        result = _session.prompt(
            HTML(f"<b>{label}</b> [{default}]: "),
            is_password=secret,
        ).strip()
        return result or default
    return _session.prompt(
        HTML(f"<b>{label}</b>: "),
        is_password=secret,
    ).strip()


def _prompt_int(label: str, *, default: int | None = None) -> int | None:
    """Prompt for an integer, returning None on blank."""
    hint = f" [{default}]" if default is not None else ""
    while True:
        raw = _session.prompt(
            HTML(f"<b>{label}</b>{hint}: "),
        ).strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            _console.print("  Please enter a number.")
