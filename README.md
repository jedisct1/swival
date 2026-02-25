![Swival Logo](.media/logo.png)

# Swival

A small, powerful, open-source CLI coding agent that works with open models.

Swival connects to [LM Studio](https://lmstudio.ai/) or
[HuggingFace Inference API](https://huggingface.co/inference-api), sends your
task, and runs an autonomous tool-calling loop until it produces an answer. No
configuration needed.

It auto-discovers your loaded model, gives it sandboxed file access, and gets out of the way.

It's what I use every day. Try it and see if you like it too.

## Quickstart

1. Install [LM Studio](https://lmstudio.ai/) and load a model with tool-calling
   support (I recommend
   [qwen3-coder-next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)).
   Crank the context size as high as your hardware allows.
2. Start the LM Studio server.
3. Install Swival:

```sh
uv tool install swival
```

4. Run:

```sh
swival "Refactor the error handling in src/api.py"
```

That's it. Swival finds the model, connects, and goes to work.

For interactive sessions:

```sh
swival --repl
```

To update to the latest version:

```sh
uv tool upgrade swival
```

To uninstall:

```sh
uv tool uninstall swival
```

## Documentation

- [Getting Started](docs.md/getting-started.md) -- installation, first run, what
  happens under the hood
- [Usage](docs.md/usage.md) -- one-shot mode, REPL mode, CLI flags, piping,
  exit codes
- [Tools](docs.md/tools.md) -- what the agent can do: file ops, search, editing,
  web fetching, thinking, command execution
- [Safety and Sandboxing](docs.md/safety-and-sandboxing.md) -- path resolution,
  symlink protection, command whitelisting, YOLO mode
- [Skills](docs.md/skills.md) -- creating and using SKILL.md-based agent skills
- [Customization](docs.md/customization.md) -- project instructions, system prompt
  overrides, tuning parameters
- [Providers](docs.md/providers.md) -- LM Studio and HuggingFace configuration
- [Reports](docs.md/reports.md) -- JSON reports for benchmarking and evaluation
- [Using Swival with AgentFS](docs.md/agentfs.md) -- copy-on-write filesystem
  sandboxing for safe agent runs
