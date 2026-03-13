# A2A

Swival can connect to remote agents via the [Agent-to-Agent (A2A) protocol](https://google.github.io/A2A/) v1.0. When A2A agents are configured, Swival fetches their Agent Cards at startup, discovers their skills, and exposes each skill as a tool in the agent loop.

Unlike MCP tools, which have custom parameter schemas defined by the server, A2A skills always use the same generic shape: a `message` string (required), plus optional `context_id` and `task_id` for multi-turn conversations. The model talks to remote agents in natural language.

Each A2A tool is namespaced as `a2a__<agent_name>__<skill_id>` to avoid collisions with built-in tools and across agents. The model calls them like any other tool. Swival routes the call to the correct agent and returns the result.

If an A2A agent fails to connect at startup, Swival logs a warning and continues without that agent's tools. If an agent fails mid-session, its tools are marked as degraded and return an error message instead of blocking the agent loop.

Tool name collisions across agents cause the colliding agent's tools to be skipped entirely, with a warning.

## Agent Card Discovery

At startup, Swival fetches each agent's Agent Card from `<url>/.well-known/agent-card.json` (or a custom `card_url` if configured). The card declares the agent's name, description, skills, and endpoint URL.

If the card declares skills, each skill becomes a separate tool. If no skills are declared, Swival creates a single generic `ask` tool for that agent.

## Multi-Turn Conversations

A2A supports multi-turn conversations through `contextId`. When an agent returns a result, the response includes a `contextId` that groups related interactions. To continue the conversation, the model passes that `contextId` back in the next call.

Some agents may return an `input-required` state, meaning they need more information before completing the task. In that case, the response includes both a `contextId` and a `taskId`. To resume, the model passes both back in the next call.

During context compaction, Swival preserves these IDs so the model can continue multi-turn interactions even after older messages are dropped.

## TOML Configuration

Add `[a2a_servers.<name>]` tables to `swival.toml`. Each agent needs a `url` pointing to its A2A endpoint.

```toml
[a2a_servers.research-agent]
url = "https://research.example.com"

[a2a_servers.code-review]
url = "https://review.example.com"
auth_type = "bearer"
auth_token = "sk-..."
timeout = 180
```

Agent names must match `[a-zA-Z0-9_-]+` and cannot contain double underscores (since `__` is used as the namespacing separator in tool names).

### Configuration Fields

| Field        | Required | Description                                                                       |
| ------------ | -------- | --------------------------------------------------------------------------------- |
| `url`        | Yes      | Base URL of the A2A agent                                                         |
| `card_url`   | No       | Override for the Agent Card URL (defaults to `<url>/.well-known/agent-card.json`) |
| `auth_type`  | No       | Authentication type: `bearer` or `api_key`                                        |
| `auth_token` | No       | Authentication token or key                                                       |
| `timeout`    | No       | Request timeout in seconds (default: 120)                                         |

### Authentication

Two authentication methods are supported:

Bearer token -- sends `Authorization: Bearer <token>` on all requests:

```toml
[a2a_servers.my-agent]
url = "https://agent.example.com"
auth_type = "bearer"
auth_token = "sk-..."
```

API key -- sends `X-API-Key: <token>` on all requests:

```toml
[a2a_servers.my-agent]
url = "https://agent.example.com"
auth_type = "api_key"
auth_token = "key-..."
```

## Standalone Config File

You can also put A2A configuration in a separate TOML file and pass it with `--a2a-config`:

```sh
swival --a2a-config agents.toml "task"
```

The file uses the same `[a2a_servers.*]` format as `swival.toml`:

```toml
[a2a_servers.research-agent]
url = "https://research.example.com"

[a2a_servers.code-review]
url = "https://review.example.com"
```

When both `--a2a-config` and `swival.toml` define agents, the TOML config takes precedence by agent name.

## Config Precedence

When the project and global config both define A2A agents, project-level agents win by name, and global-only agents are merged in.

`--no-a2a` disables A2A agent connections entirely, even if agents are configured.

## CLI Flags

`--no-a2a` disables A2A agent connections entirely.

`--a2a-config FILE` provides an explicit path to an A2A TOML config file.

## Output Handling

A2A tool outputs are size-guarded the same way as MCP tools. Results up to 20 KB are returned inline. Larger results are saved to `.swival/cmd_output_*.txt` and the model receives a pointer message to use `read_file` for paginated access.

When saving large A2A output to file, Swival preserves continuation metadata (the `[contextId=...]` or `[input-required]` header line) so the model can still continue multi-turn conversations even when the response body is too large for inline display.

Error outputs are kept inline but truncated at 20 KB.

During context compaction, A2A tool results receive structured summaries that preserve `input-required` headers with their `contextId` and `taskId`, so multi-turn state survives compaction.

## Library API

The `Session` class accepts `a2a_servers` as a constructor argument. Pass a dictionary mapping agent names to config dicts:

```python
from swival import Session

session = Session(
    a2a_servers={
        "research-agent": {
            "url": "https://research.example.com",
            "auth_type": "bearer",
            "auth_token": "sk-...",
        }
    }
)
result = session.run("Ask the research agent to summarize recent papers on RAG")
```

Use `Session` as a context manager to ensure A2A connections are cleaned up:

```python
with Session(a2a_servers={"agent": {"url": "https://..."}}) as session:
    result = session.run("task")
```

## Protocol Details

Swival implements the A2A v1.0 JSON-RPC binding. It sends `SendMessage` requests with `returnImmediately=false` (blocking mode) as the primary path. If the server returns a non-terminal task instead of blocking, Swival falls back to polling with `GetTask` using exponential backoff.

Responses can be either Task-shaped (with `id`, `status`, and optional `artifacts`) or Message-shaped (with `role` and `parts`). Swival handles both.

Task states are categorized as:

- Terminal: `completed`, `failed`, `canceled`, `rejected`
- Interrupted: `input-required`, `auth-required`

Terminal tasks return their result immediately. Interrupted tasks return with continuation metadata so the model can resume.
