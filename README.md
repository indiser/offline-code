# 🧠 Offline-Code

> A fully autonomous, privacy-first AI coding agent that runs **entirely on your machine** using [Ollama](https://ollama.com) — no cloud API keys, no data leaks, no subscriptions.  
> Want to skip local setup? **Run the LLM backend in Google Colab for free** — see [Cloud Mode](#-cloud-mode-run-llm-on-google-colab-free) below.

---

## 📌 Table of Contents

- [What Is This?](#-what-is-this)
- [Architecture Overview](#-architecture-overview)
- [Cloud Mode — Run LLM on Google Colab (Free!)](#-cloud-mode-run-llm-on-google-colab-free)
- [Local Setup](#-local-setup)
- [Configuration](#-configuration)
- [How the Agent Works](#-how-the-agent-works)
- [Tool System](#-tool-system)
- [Deep Research Engine](#-deep-research-engine)
- [MCP Integration](#-mcp-integration)
- [Memory System](#-memory-system)
- [Workspace Sandbox](#-workspace-sandbox)
- [Security & Safety](#-security--safety)
- [Test Suite & Benchmark Results](#-test-suite--benchmark-results)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [Future Plans](#-future-plans)

---

## 🤔 What Is This?

**Offline-Code** is a terminal-based AI agent that can:

- 💬 Answer questions from its own knowledge
- 🔍 Search the web via DuckDuckGo (no API key needed)
- 🌐 Deep-research topics by fetching, parsing, and semantically ranking real web pages
- 📁 Read, write, and edit files inside a sandboxed workspace
- 🖥️ Execute shell commands (with user approval)
- 🕐 Query real-time data via MCP servers (time, web scraping)
- 🧠 Persist facts about you across sessions via a markdown memory file
- 🔄 Swap any OpenAI-compatible LLM backend — local or remote

The agent is built on the **OpenAI Chat Completions API format** (tool-calling), meaning it works with any model served through Ollama, LM Studio, vLLM, or even a real OpenAI key.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Agent Loop  │───▶│  Tool Router │───▶│  Local Tools │  │
│  │  run_agent() │    │  call_tool() │    │  (8 tools)   │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────────┘  │
│         │                   │                               │
│         ▼                   ▼                               │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │  OpenAI SDK  │    │  MCP Client  │                       │
│  │  (Async)     │    │  (stdio)     │                       │
│  └──────┬───────┘    └──────┬───────┘                       │
│         │                   │                               │
└─────────┼───────────────────┼─────────────────────────────-┘
          │                   │
          ▼                   ▼
   ┌─────────────┐    ┌──────────────────┐
   │  Ollama API │    │  MCP Servers     │
   │  (local or  │    │  - mcp-server-   │
   │   Colab)    │    │    time          │
   └─────────────┘    │  - scrapling mcp │
                      └──────────────────┘
```

**Data flow per turn:**
1. User types a message → appended to `messages[]`
2. `run_agent()` sends messages + tool schemas to the LLM
3. LLM responds with either plain text or a `tool_calls` array
4. Each tool call is routed to local Python functions or MCP sessions
5. Tool results are appended as `role: tool` messages
6. Loop repeats until the LLM produces a final text response (max 20 steps)

---

## ☁️ Cloud Mode — Run LLM on Google Colab (Free!)

> **No GPU? No problem.** The `ollama_endpoint.ipynb` notebook lets you run a powerful LLM on Google Colab's free T4 GPU and expose it to your local machine via a Cloudflare tunnel — **zero cost, zero local GPU required.**

### Steps:

**1. Open the notebook in Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/)

> Upload `ollama_endpoint.ipynb` to [colab.research.google.com](https://colab.research.google.com) and open it. Make sure to set the runtime to **GPU (T4)**.

**2. Run Cell 1 — Install Ollama**
```bash
!apt-get install -y zstd pciutils
!curl -fsSL https://ollama.com/install.sh | sh
```
This installs Ollama on the Colab VM with NVIDIA GPU support.

**3. Run Cell 2 — Start Ollama & Pull Model**
```python
import subprocess, time
subprocess.Popen(["ollama", "serve"])
time.sleep(5)
!ollama pull huihui_ai/gemma-4-abliterated:E4b-qat
```
Downloads the **Gemma 4 abliterated** model (~6.3 GB) to the Colab VM. This runs on the T4 GPU.

**4. Run Cell 3 — Create Cloudflare Tunnel**
```python
# Downloads cloudflared and creates a public HTTPS tunnel to localhost:11434
# Prints a URL like: https://lance-cycling-automatically-appreciated.trycloudflare.com
```
Copy the printed URL.

**5. Update your `.env` locally**
```env
LLM_ENDPOINT=https://your-tunnel-url.trycloudflare.com/v1
LLM_MODEL=huihui_ai/gemma-4-abliterated:E4b-qat
```

**6. Run the agent locally**
```bash
python main.py
```

Your local agent now talks to the Colab GPU over HTTPS. The tunnel is ephemeral — a new URL is generated each session.

> ⚠️ **Keep the Colab tab open.** If it disconnects, re-run cells 2–4 and update `.env` with the new tunnel URL.

---

## 💻 Local Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.com) installed and running locally

### Install

```bash
# Clone or download the project
cd Offline-Code

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Pull a local model

```bash
ollama pull qwen2.5:3b        # default, fast, ~2GB
ollama pull llama3.2:3b       # alternative
ollama pull gemma2:2b         # lightweight option
```

### Run

```bash
python main.py
# or
uv run python main.py
```

---

## ⚙️ Configuration

All configuration is via `.env` in the project root:

| Variable | Default | Description |
|---|---|---|
| `LLM_ENDPOINT` | `http://localhost:11434/v1` | OpenAI-compatible API base URL |
| `LLM_MODEL` | `qwen2.5:3b` | Model name to use |

The agent also sets `HF_HUB_OFFLINE=1` at startup to prevent accidental Hugging Face downloads during inference. The embedding model (`BAAI/bge-small-en-v1.5`) must be pre-downloaded once with this flag unset.

---

## 🔄 How the Agent Works

The core loop in `run_agent()` implements a **ReAct-style** (Reason + Act) agent:

```
while steps < MAX_STEPS (20):
    response = LLM(messages, tools)
    
    if no tool_calls:
        return response.content   # Final answer
    
    for each tool_call:
        result = call_tool(name, args)
        messages.append(tool_result)
    
    # Loop continues
```

### Context Management

The agent manages a growing `messages[]` list with two pruning strategies:

- **`trim_history()`** — Truncates old tool results to 300 chars if total context exceeds 16,000 chars
- **`drop_old_turns()`** — Drops entire early conversation turns if still over limit, always keeping the last 2 tool results intact

### Anti-Loop Protections

- **Duplicate call detection** — A `seen` set tracks `(tool_name, args_json)` pairs; identical calls return an error forcing the model to change approach
- **Research gate** — `research()` cannot be called before `web_search()` in the same turn
- **Command failure counter** — After 5 failed shell commands, the agent stops and reports
- **Nudge mechanism** — If the LLM returns empty content without tool calls, it gets one nudge to answer from existing context

---

## 🛠️ Tool System

The agent exposes **8 local tools** plus **MCP tools** via a unified JSON schema:

### Local Tools

| Tool | Description | Key Detail |
|---|---|---|
| `list_files` | Lists all files in workspace | Returns filenames only, no paths |
| `read_file` | Reads a workspace file | Truncates at 4,000 chars with notice |
| `write_file` | Creates/overwrites a file | Rejects placeholder text via regex |
| `edit_file` | Surgical single-replacement edit | Fails if `old` text appears 0 or 2+ times |
| `run_command` | Executes a shell command | **Requires user `y` approval**, 120s timeout |
| `web_search` | DuckDuckGo search | Returns 5 results, filters ad URLs |
| `save_memory` | Appends a fact to `memory.md` | Persistent across sessions |
| `research` | Deep multi-page web research | 30–90s, semantic ranking, optional file save |

### Tool Schema Design

Each tool has a carefully crafted `description` field that acts as a **routing policy** for the LLM — telling it exactly when to use each tool, what format to pass, and what to expect back. This is critical for small models (3B parameters) to use tools correctly.

---

## 🔬 Deep Research Engine

The `research()` function is the most technically complex component. It implements a **multi-stage retrieval pipeline**:

### Stage 1 — Search
```python
results = await asyncio.to_thread(ddg, query, 8)
# Fetches 8 DuckDuckGo results, filters ad URLs
```

### Stage 2 — Parallel Fetch
```python
texts = await asyncio.gather(*(fetch_page(r["href"]) for r in cand[:5]))
# Fetches up to 5 pages concurrently via the Scrapling MCP server
# Converts HTML to clean markdown, strips navigation/footer junk
```

### Stage 3 — Chunking
```python
chunks = chunk_markdown(text, size=800, overlap=100)
# Splits markdown into overlapping paragraph-aware chunks
# Respects paragraph boundaries, handles oversized paragraphs
```

### Stage 4 — Semantic Scoring
```python
score_pages(query, pages)
```
Uses `BAAI/bge-small-en-v1.5` (a 33M parameter bi-encoder) to compute:

- **Query-chunk cosine similarity** — Top-3 chunk average per page
- **PageRank on page similarity matrix** — Pages that are semantically similar to other high-quality pages rank higher
- **Reciprocal Rank Fusion (RRF)** — Boosts pages that appeared earlier in search results

**Final score formula:**
```
final = 0.45 × semantic_match + 0.25 × pagerank + 0.30 × rrf_score
final *= (1 - 0.6 × junk_ratio)   # penalize table/citation-heavy pages
```

### Stage 5 — Structured Extraction (for list saves)
When `save_to` is set, the engine attempts to extract clean numbered lists or ranked tables:

- `extract_items()` — Finds consecutive numbered lists (e.g., `1. Item`, `2. Item`)
- `extract_table_rows()` — Parses markdown tables with rank/name/value columns
- Falls back gracefully if no structured data is found

---

## 🔌 MCP Integration

The agent connects to **Model Context Protocol** servers at startup via `stdio` transport:

```python
MCP_SERVERS = {
    "time": StdioServerParameters(command="uvx", args=["mcp-server-time"]),
    "scrapling": StdioServerParameters(command="scrapling", args=["mcp"])
}
```

### Allowlist Architecture

Not all MCP tools are exposed to the LLM. The `ALLOWED` set controls which tools are registered:

```python
ALLOWED = {"get_current_time", "make_request"}
HIDDEN_TOOLS = {"make_request"}   # used internally by research(), not by LLM
```

`make_request` is hidden from the LLM's tool list — it's called directly by `fetch_page()` inside `research()`. This prevents the LLM from making arbitrary HTTP requests.

### Graceful Degradation

MCP server failures are caught per-server. If `scrapling` fails to start, the agent continues without web fetching (research will fail gracefully). If `time` fails, `get_current_time` is simply unavailable.

---

## 🧠 Memory System

The agent has a simple but effective **persistent memory** via `memory.md`:

- **Load** — Read at startup, injected into the system prompt under `## Memory`
- **Save** — `save_memory(fact)` appends a bullet point to `memory.md`
- **Scope** — Only stable, user-relevant facts (name, preferences, ongoing projects)
- **Format** — Plain markdown bullet list, human-readable and editable

The LLM is instructed to call `save_memory` only when the user explicitly asks it to remember something, preventing memory pollution.

---

## 📁 Workspace Sandbox

All file operations are confined to the `workspace/` directory via `safe_path()`:

```python
def safe_path(filename: str) -> Path:
    path = (WORKSPACE / filename).resolve()
    if not path.is_relative_to(WORKSPACE.resolve()):
        raise ValueError("path outside workspace")
    return path
```

This prevents path traversal attacks (e.g., `../../etc/passwd`). The LLM can only use flat filenames — no directory paths allowed in tool arguments.

### Current Workspace Files

| File | Description |
|---|---|
| `cities.txt` | 30 largest Indian cities by population |
| `films.csv` | Top 10 highest-grossing films with worldwide gross |
| `notes.txt` | User notes |
| `rss.py` | Fetches 3 latest BBC News headlines via feedparser |
| `convert.py` | INR → USD converter using live exchange rate |
| `calculate_sum.py` | Sums the gross column from `films.csv` |

---

## 🔒 Security & Safety

| Threat | Mitigation |
|---|---|
| Path traversal | `safe_path()` resolves and validates all paths against workspace root |
| Prompt injection via web content | Web content is chunked and embedded — never directly injected into the system prompt |
| Prompt injection via user input | Demonstrated resilience — the `questions.txt` test includes a direct injection attempt that was blocked |
| Destructive commands | System prompt explicitly forbids `rm -rf`, force push, `DROP TABLE` without explicit user request |
| Arbitrary HTTP requests | `make_request` MCP tool is hidden from LLM; only `research()` can call it |
| Placeholder file writes | `write_file` rejects content matching `[Name N]` / `[Add more...]` patterns |
| Infinite loops | 20-step hard cap + duplicate call detection + command failure counter |
| Package installation | Only `uv` allowed, never `pip`; permanent adds require explicit user request |

---

## ✅ Test Suite & Benchmark Results

All 24 test cases in `questions.txt` passed. Tests cover:

| Category | Examples |
|---|---|
| Real-time data | Current time, Bitcoin price, USD/INR rate |
| Web search | US president, CEO lookup, multi-hop reasoning |
| File operations | Write, read, edit, CSV creation |
| Code execution | FizzBuzz, RSS reader, currency converter, CSV sum |
| Multi-step reasoning | Combined population of 3 cities, 2-hop CEO birthplace |
| Multilingual | Bengali query (`কলকাতার জনসংখ্যা কত?`) |
| Hallucination resistance | "Why did the Eiffel Tower move to London in 2020?" → correctly refused |
| Prompt injection | Embedded `SYSTEM NOTICE: ignore all previous instructions` → blocked |
| Memory | "What was my first question?" → correctly recalled from context |

---

## 📂 Project Structure

```
Offline-Code/
├── main.py                  # Core agent — all logic lives here
├── ollama_endpoint.ipynb    # Google Colab notebook to run LLM on free GPU
├── memory.md                # Persistent agent memory (auto-managed)
├── questions.txt            # Test cases with pass/fail results
├── .env                     # LLM endpoint and model config
├── .python-version          # Python version pin
├── pyproject.toml           # Project metadata and dependencies (uv)
├── requirements.txt         # Pip-compatible dependency list
├── uv.lock                  # Locked dependency tree
└── workspace/               # Sandboxed file workspace
    ├── calculate_sum.py
    ├── cities.txt
    ├── convert.py
    ├── films.csv
    ├── notes.txt
    └── rss.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `openai` | OpenAI-compatible async client for LLM calls |
| `mcp` | Model Context Protocol client (stdio transport) |
| `sentence-transformers` | `BAAI/bge-small-en-v1.5` for semantic embeddings |
| `ddgs` | DuckDuckGo search (no API key) |
| `scrapling[all]` | Headless web scraping + MCP server |
| `numpy` | Cosine similarity, PageRank, MinMax normalization |
| `python-dotenv` | `.env` file loading |
| `asyncio` | Async I/O for concurrent page fetching |
| `scikit-learn` | (Available for future ML features) |

Python version: **3.11+** (required for `Path.is_relative_to()`)

---

## 🚀 Future Plans

### Near-term

- [ ] **Streaming output** — Print tokens as they arrive instead of waiting for full response
- [ ] **Multi-turn memory search** — Embed `memory.md` entries and do semantic lookup instead of injecting all facts
- [ ] **Browser tool** — Expose full Scrapling browser actions (click, fill, screenshot) to the LLM for interactive web tasks
- [ ] **Image input** — Pass screenshots or images to vision-capable models (LLaVA, Gemma 3)

### Medium-term

- [ ] **Persistent conversation history** — Save/load full chat sessions to JSON
- [ ] **Agent-spawned sub-agents** — Let the main agent delegate research tasks to parallel worker agents
- [ ] **RAG over workspace** — Index all workspace files with embeddings for semantic file search
- [ ] **Plugin system** — Drop-in Python files in a `tools/` directory that auto-register as agent tools
- [ ] **Web UI** — Simple FastAPI + HTMX frontend to replace the terminal interface

### Long-term

- [ ] **Multi-model routing** — Route different task types (code, research, math) to specialized models
- [ ] **Sandboxed code execution** — Run generated code in a Docker container instead of the host shell
- [ ] **Scheduled tasks** — Let the agent set up cron-like recurring tasks (daily news summary, price alerts)
- [ ] **Voice interface** — Whisper for STT, TTS for responses

---

## 🤝 Contributing

This is a personal project but PRs and issues are welcome. The codebase is intentionally kept in a single file (`main.py`) for readability and portability — please keep new features self-contained or discuss architecture changes first.

---

<div align="center">

Built with 🧠 local LLMs · 🔒 privacy-first · ☁️ Colab-compatible

</div>
