# Web Browsing

Swival can browse the web, interact with pages, fill forms, take screenshots, and extract content. There are three ways to set this up, each with different tradeoffs.

| Approach | How it works | Best for |
|---|---|---|
| [Chrome DevTools MCP](#chrome-devtools-mcp) | MCP server controlling Chrome via Puppeteer | Full browser fidelity, debugging, performance profiling |
| [agent-browser](#agent-browser) | CLI tool called via `run_command` | Fast snapshots, low token usage, headless automation |
| [agent-browser + Lightpanda](#lightpanda) | Same CLI, different engine | Speed, low memory, CI/CD pipelines |

## Chrome DevTools MCP

[Chrome DevTools MCP](https://github.com/ChromeDevTools/chrome-devtools-mcp) is Google's official MCP server for browser automation. It gives Swival access to the full Chrome DevTools feature set, including navigation, clicking, form filling, screenshots, console messages, network inspection, and performance tracing.

### Setup

Add it to your `swival.toml`:

```toml
[mcp_servers.chrome]
command = "npx"
args = ["-y", "chrome-devtools-mcp@latest"]
```

This launches Chrome with a visible window. For headless operation (no UI), add `--headless`:

```toml
[mcp_servers.chrome]
command = "npx"
args = ["-y", "chrome-devtools-mcp@latest", "--headless"]
```

Requires Node.js v20.19+ and Chrome (stable channel).

### What it gives Swival

Once configured, Swival gets MCP tools like `mcp__chrome__navigate_page`, `mcp__chrome__click`, `mcp__chrome__fill`, `mcp__chrome__take_screenshot`, `mcp__chrome__list_console_messages`, and more. The model calls them like any other tool.

### Example

```sh
swival --repl
> Navigate to https://news.ycombinator.com, find the top story, and summarize it
```

Swival will call `navigate_page`, then `snapshot` or `get_page_content` to read the page, and return a summary.

### Headless with isolation

For throwaway sessions that leave no browser state behind:

```toml
[mcp_servers.chrome]
command = "npx"
args = ["-y", "chrome-devtools-mcp@latest", "--headless", "--isolated"]
```

The `--isolated` flag creates a temporary profile that gets cleaned up when the session ends.

### Slim mode

If you only need basic navigation and don't want the full DevTools toolset (performance, emulation, etc.), use `--slim`:

```toml
[mcp_servers.chrome]
command = "npx"
args = ["-y", "chrome-devtools-mcp@latest", "--headless", "--slim"]
```

This reduces the number of tools exposed to the model, which saves context window space.

## agent-browser

[agent-browser](https://github.com/vercel-labs/agent-browser) is a CLI by Vercel Labs that controls a browser from the command line. It produces compact text output optimized for AI agents and uses reference-based element selection that burns fewer tokens than DOM selectors.

Instead of connecting via MCP, agent-browser works through Swival's `run_command` tool. You whitelist the `agent-browser` command and the model calls it directly.

### Install

```sh
npm install -g agent-browser
agent-browser install        # downloads Chrome for Testing (first time only)
```

Or with Homebrew on macOS:

```sh
brew install agent-browser
agent-browser install
```

### Configure Swival

Allow the `agent-browser` command in `swival.toml`:

```toml
allowed_commands = ["agent-browser"]
```

Or pass it on the command line:

```sh
swival --allowed-commands agent-browser "Open example.com and tell me what's on the page"
```

### How the model uses it

The model calls `run_command` with agent-browser subcommands. First it opens a URL, then takes a snapshot to get an accessibility tree with element refs like `@e1` and `@e2`, then interacts with those elements, and finally closes the browser when it's done:

```sh
agent-browser open <url>
agent-browser snapshot -i
agent-browser click @e1
agent-browser fill @e2 "search query"
agent-browser screenshot page.png
agent-browser close
```

The snapshot output looks like this:

```
- none
  - heading "Example Domain" [ref=e1]
    - StaticText "Example Domain"
  - paragraph
    - StaticText "This domain is for use in..."
  - paragraph
    - link "Learn more" [ref=e2]
      - StaticText "Learn more"
```

This is 200-400 tokens compared to 3,000-5,000 for a full DOM dump, which matters when you're browsing multiple pages in a single session.

### Example session

```sh
swival --allowed-commands agent-browser --repl
> Go to https://news.ycombinator.com, click on the top story, and summarize the article
```

The model will run `agent-browser open`, `agent-browser snapshot -i`, `agent-browser click @e1`, and so on, chaining commands until it has the information it needs.

### More commands

agent-browser has 50+ commands. Some useful ones:

```sh
agent-browser get text @e1          # get text content of an element
agent-browser get title             # page title
agent-browser get url               # current URL
agent-browser tab new https://...   # open a new tab
agent-browser tab 2                 # switch tabs
agent-browser scroll down 500       # scroll the page
agent-browser eval "document.title" # run JavaScript
agent-browser pdf report.pdf        # save page as PDF
```

## Lightpanda

[Lightpanda](https://lightpanda.io/) is a headless browser written from scratch in Zig. It skips pixel rendering entirely and focuses on DOM processing and JavaScript execution via V8. The result is roughly 10x faster execution and 10x less memory than headless Chrome.

Lightpanda works with agent-browser as a drop-in engine replacement. Same commands, same output format, different browser underneath.

### Why use Lightpanda instead of Chrome

Lightpanda starts instantly and loads pages faster because there's no rendering pipeline. Each instance uses about 24 MB of memory compared to Chrome's 207 MB, which makes it lightweight enough to run in constrained CI/CD environments or to spin up many instances in parallel for concurrent scraping.

The tradeoff is that Lightpanda is still in beta. Most websites work, but you may hit gaps in Web API coverage. It doesn't support screenshots, extensions, or persistent browser profiles either. Use Chrome when you need full fidelity.

### Install Lightpanda

Download the binary for your platform:

**macOS (Apple Silicon):**

```sh
curl -L -o lightpanda https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-aarch64-macos
chmod a+x ./lightpanda
sudo mv ./lightpanda /usr/local/bin/
```

**Linux (x86_64):**

```sh
curl -L -o lightpanda https://github.com/lightpanda-io/browser/releases/download/nightly/lightpanda-x86_64-linux
chmod a+x ./lightpanda
sudo mv ./lightpanda /usr/local/bin/
```

Verify it works:

```sh
lightpanda fetch --dump html https://example.com
```

### Use with agent-browser

Set the `AGENT_BROWSER_ENGINE` environment variable so agent-browser uses Lightpanda instead of Chrome:

```sh
export AGENT_BROWSER_ENGINE=lightpanda
```

That's it. All agent-browser commands work the same way and the engine swap is transparent:

```sh
agent-browser open https://example.com
agent-browser snapshot -i
agent-browser close
```

You can also pass the engine per-command if you don't want to set it globally:

```sh
agent-browser --engine lightpanda open https://example.com
```

Or put it in an `agent-browser.json` config file in your project directory:

```json
{
  "engine": "lightpanda"
}
```

### Configure Swival with Lightpanda

Add the environment variable to your shell profile (`.zshrc`, `.bashrc`, etc.) and configure Swival the same way as regular agent-browser:

```toml
allowed_commands = ["agent-browser"]
```

Then run:

```sh
export AGENT_BROWSER_ENGINE=lightpanda
swival --allowed-commands agent-browser "Open example.com and describe the page"
```

### Lightpanda as an MCP server

Lightpanda also has a built-in MCP server mode that you can connect to Swival directly, without agent-browser:

```toml
[mcp_servers.lightpanda]
command = "lightpanda"
args = ["mcp"]
```

This exposes Lightpanda's browsing tools through MCP. It has fewer tools than Chrome DevTools MCP, but starts faster and uses far less resources.

### Disabling telemetry

Lightpanda collects usage telemetry by default. To disable it:

```sh
export LIGHTPANDA_DISABLE_TELEMETRY=true
```

## Which approach should I use?

Pick **Chrome DevTools MCP** if you need the full browser, including screenshots, network inspection, performance profiling, extensions, or sites that require complete rendering fidelity.

Pick **agent-browser with Chrome** if you want a simpler setup with lower token usage and don't need MCP-level tool integration. It works well for straightforward browsing, form filling, and content extraction.

Pick **agent-browser with Lightpanda** if speed and resource efficiency matter more than full browser compatibility. It's the best option for scraping, CI pipelines, and tasks where you're hitting many pages in sequence.

You can also combine approaches. For example, use Chrome DevTools MCP for debugging your web app and agent-browser with Lightpanda for bulk research tasks.
