import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding= "utf-8")
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import asyncio
from contextlib import AsyncExitStack
import json
from pathlib import Path
import subprocess
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from ddgs import DDGS
from datetime import date
import time
import re
import numpy as np
import traceback
from dotenv import load_dotenv

load_dotenv()

# Just tools name for the future no need to change anything.
# SCAPLING_MCP_TOOLS_LIST = ['browser_open', 'browser_save_state', 'browser_load_state', 'open_request_session', 'close_session', 'list_sessions', 'make_request', 'browser_fetch_once', 'browser_fetch', 'session_make_request', 'browser_extract', 'browser_actions', 'browser_evaluate', 'browser_network_requests', 'browser_network_request', 'browser_screenshot']


client = AsyncOpenAI(
    base_url = os.environ.get("LLM_ENDPOINT", "http://localhost:11434/v1"),
    api_key = "ollama"
)
MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")

WORKSPACE = Path(__file__).parent/"workspace"
os.makedirs(WORKSPACE, exist_ok=True)

MEMORY_FILE = Path(__file__).parent/"memory.md"


def safe_path(filename: str) -> Path:
    path = (WORKSPACE / filename).resolve()
    if not path.is_relative_to(WORKSPACE.resolve()):
        raise ValueError("path outside workspace")
    return path

# Load the agent memory
def load_memory() -> str:
    if MEMORY_FILE.is_file():
        return MEMORY_FILE.read_text(encoding="utf-8")
    return "(nothing saved yet)"

# Save agents memory
def save_memory(fact: str) -> str:
    with MEMORY_FILE.open("a", encoding="utf-8") as filp:
        filp.write(f"- {fact}\n")
    return f"saved: {fact}"

MCP_SERVERS = {
    "time": StdioServerParameters(command="uvx", args = ["mcp-server-time"]),
    "scrapling": StdioServerParameters(command="scrapling", args=["mcp"])
}

# File Look Up
def list_files() -> str:
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"

# Read Files
def read_file(filename: str) -> str:
    path = safe_path(filename)
    if not path.is_file():
        return f"error: no file named {filename}"
    text = path.read_text(encoding="utf-8")
    if len(text) > 4000:
        return text[:4000] + f"\n[truncated, file has {len(text)} chars]"
    return text

# Write And Modify files
def write_file(filename: str, contents: str) -> str:
    if re.search(r"\[(?:Name \d+|Add more[^\]]*|List of[^\]]*)\]", contents, re.I):
        return "error: contents has placeholder text. Write only real content."
    safe_path(filename=filename).write_text(contents, encoding="utf-8")
    return f"wrote: {filename}"

# Edit a File
def edit_file(filename: str, old: str, new: str) -> str:
    path = safe_path(filename)
    if not path.is_file():
        return f"error: no file named {filename}"
    text = path.read_text(encoding="utf-8")
    n = text.count(old)
    if n == 0:
        return "error: old text not found. read_file first, copy exact text."
    if n > 1:
        return f"error: old text found {n} times. Include more surrounding lines."
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"edited: {filename}"

VENV = Path(__file__).parent / ".venv"

def venv_env() -> dict:
    if not VENV.exists():
        subprocess.run(["uv", "venv", "--seed", str(VENV)], check=True, capture_output=True)
    bindir = VENV / ("Scripts" if os.name == "nt" else "bin")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}
    env["VIRTUAL_ENV"] = str(VENV)
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    return env

# Run commands in the terminal
def run_command(command: str):
    if re.search(r"python\S*\s+-c\s.*;\s*(for|with|if|while|def)\b", command):
        return "error: python -c cannot hold for/with/if on one line. write_file a .py script, then run python name.py"
    answer = input(f"\n Run '{command}'? [y/N]")
    if answer.strip().lower() != 'y':
        return "The user declined to run this command"
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text = True, timeout = 120, cwd= WORKSPACE, encoding="utf-8", errors="replace", env=venv_env()
        )
    except subprocess.TimeoutExpired:
        return "error: command timed out after 120s"
    except subprocess.CalledProcessError as e:
        return f"error: venv create failed: {e}"
    
    output = (result.stdout + result.stderr).strip()
    print(f"\n [Command Result] {output[-2000:]}")
    return output[-3000:] or f"(no output, exit code {result.returncode})"

# Helper fuction for Duck duck go search
def ddg(query: str, n: int) -> list[dict]:
    last = None
    for _ in range(2):
        try:
            rs = DDGS().text(query, max_results=n + 3) or []
            rs = [r for r in rs
                  if "bing.com/aclick" not in r.get("href", "")
                  and "duckduckgo.com/y.js" not in r.get("href", "")]
            return rs[:n]
        except Exception as e:
            last = e
            time.sleep(1.5)
    raise last

# Search The Web using deck duck go
def web_search(query: str) -> str:
    try:
        results = ddg(query, 5)
    except Exception as e:
        return f"error: search failed twice: {e}"
    return "\n\n".join(
        f"{r['title']}\n{r['href']}\n{r['body']}" for r in results
    ) or "(no results)"

EMBED_PATH = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_embedder = None

# Embedding function
def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        try:
            _embedder = SentenceTransformer(EMBED_PATH, device="cpu")
        except Exception as e:
            raise RuntimeError(
                f"embedding model '{EMBED_PATH}' is not in the local cache. "
                f"Download it once with HF_HUB_OFFLINE unset. ({type(e).__name__})"
            ) from None
    return _embedder

# Webpage markdown cleanup
def clean_markdown(text: str) -> str:
    # unwrap JSON if the tool returned JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "content" in data:
            data = data["content"]
        if isinstance(data, list):
            text = "\n".join(str(x) for x in data)
        elif isinstance(data, str):
            text = data
    except (json.JSONDecodeError, TypeError):
        pass
    if "\\n" in text:
        text = text.replace("\\n", "\n").replace('\\"', '"')
    text = re.sub(r"\[\[\d+\]\]\(#[^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# Webpage markdown format analyze in chunks
def chunk_markdown(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
            while len(p) > size:
                chunks.append(p[:size])
                p = p[size - overlap:]
            cur = p
    if cur:
        chunks.append(cur)
    return chunks

# Extract items from the markdown
def extract_items(text: str, min_run: int = 5) -> list[str]:
    runs, cur, expect = [], [], 1
    for line in clean_markdown(text).splitlines():
        m = re.match(r"^[#*\s]*(\d+)[\.\):\-]\s*(.{2,80})$", line)
        if not m or "↑" in line or "Retrieved" in line:
            continue
        n = int(m.group(1))
        name = re.sub(r"[*_`]", "", m.group(2)).strip()
        if n == expect:
            cur.append(name)
            expect += 1
        else:
            if len(cur) >= min_run:
                runs.append(cur)
            cur, expect = ([name], 2) if n == 1 else ([], 1)
    if len(cur) >= min_run:
        runs.append(cur)
    return max(runs, key=len) if runs else []

# Function to extract tables rows from a page
def extract_table_rows(text: str, min_run: int = 5) -> list[str]:
    runs, cur, expect = [], [], 1
    for line in clean_markdown(text).splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [re.sub(r"\[[^\]]*\]", "", c).strip()
                 for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue

        name = next((c for c in cells[1:] if re.search(r"[A-Za-z]", c)), "")
        pop = next((c for c in cells[1:]
                    if re.fullmatch(r"\$?\d{1,3}(?:,\d{3})+|\$?\d{6,}", c)), "")
        if not name or not pop:
            continue
        n = int(cells[0])
        item = f"{name} - {pop}"
        if n == expect:
            cur.append(item)
            expect += 1
        else:
            if len(cur) >= min_run:
                runs.append(cur)
            cur, expect = ([item], 2) if n == 1 else ([], 1)
    if len(cur) >= min_run:
        runs.append(cur)
    return max(runs, key=len) if runs else []

# MinMax algo
def minmax(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    rng = a.max() - a.min()
    return np.zeros_like(a) if rng < 1e-9 else (a - a.min()) / rng

# Simplistic Pageranck algo for ranking pages
def pagerank(sim: np.ndarray, damping: float = 0.85, iters: int = 50) -> np.ndarray:
    n = len(sim)
    if n == 1:
        return np.ones(1)
    off = sim[~np.eye(n, dtype=bool)]
    w = np.clip(sim - np.median(off), 0, None)
    np.fill_diagonal(w, 0)
    col = w.sum(axis=0, keepdims=True)
    col[col == 0] = 1
    m = w / col
    r = np.ones(n) / n
    for _ in range(iters):
        r = (1 - damping) / n + damping * (m @ r)
    return r

# Bad stuff on webstie
def junk_ratio(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 1.0
    bad = sum(1 for l in lines if l.startswith("|") or re.match(r"^\d+\.\s*[↑^]", l))
    return bad / len(lines)

# Score the webpage
def score_pages(query: str, pages: list[dict]) -> None:
    m = get_embedder()
    q = m.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    best_sim, page_vecs = [], []
    for p in pages:
        v = m.encode(p["chunks"], normalize_embeddings=True, batch_size=16)
        s = v @ q
        p["chunk_scores"] = s
        best_sim.append(float(np.sort(s)[::-1][:3].mean()))
        mv = v.mean(axis=0)
        page_vecs.append(mv / (np.linalg.norm(mv) + 1e-9))
    pv = np.array(page_vecs)
    pr = pagerank(pv @ pv.T)
    rrf = np.array([1.0 / (60 + p["rank"]) for p in pages])
    final = 0.45 * minmax(best_sim) + 0.25 * minmax(pr) + 0.30 * minmax(rrf)
    for p, f, bs, r_ in zip(pages, final, best_sim, pr):
        p["final"] = float(f * (1 - 0.6 * junk_ratio(p["text"])))
        p["match"], p["pr"] = bs, float(r_)

# Turn the webpage into markdown
async def fetch_page(url: str) -> str:
    try:
        res = await mcp_session["make_request"].call_tool("make_request", {"url": url})
        text = "\n".join(c.text for c in res.content if hasattr(c, "text"))
        return clean_markdown(text)[:150000]
    except Exception:
        return ""

# Complete deep research on the web
async def research(query: str, save_to: str = "", top_k: int = 3) -> str:
    try:
        results = await asyncio.to_thread(ddg, query, 8)
    except Exception as e:
        return f"error: search failed: {e}"
    cand = (results or [])[:5]
    if not cand:
        return "error: no search results"

    texts = await asyncio.gather(*(fetch_page(r["href"]) for r in cand))
    pages = []
    for rank, (r, t) in enumerate(zip(cand, texts)):
        if len(t) < 300:
            continue
        chunks = chunk_markdown(t)[:120]
        if chunks:
            pages.append({"url": r["href"], "title": r.get("title", ""), "rank": rank,
                          "text": t, "chunks": chunks})
    if not pages:
        return "error: no page returned usable text"

    await asyncio.to_thread(score_pages, query, pages)
    pages.sort(key=lambda p: -p["final"])
    for p in pages:
        print(f" [research] final={p['final']:.2f} match={p['match']:.2f} "
              f"pr={p['pr']:.2f} rank={p['rank']} {p['url']}")
    
    mt = re.search(r"\b(?:top|best)\s+(\d+)|\b(\d+)\s+(?:largest|biggest|top|best|most)", query, re.I)
    want = int(mt.group(1) or mt.group(2)) if mt else 10
    need = max(5, int(want * 0.8))
    if save_to:
        for p in pages:
            items = next((c for c in (extract_table_rows(p["text"]), extract_items(p["text"]))
                          if len(c) >= need), [])
            if not items:
                rows = [l[:250] for l in p["text"].splitlines() if l.lstrip().startswith("|")]
                print(f" [debug] {p['url']} len={len(p['text'])} table_lines={len(rows)}")
                for r in rows[:6]:
                    print(" [debug]", r)
            if save_to.lower().endswith(".csv") and not all(
                    re.search(r"\d{1,3}(?:,\d{3}){2,}|\d{7,}", it) for it in items):
                items = []
            if mt:
                items = items[:want]
            print(f" [research] {len(items)} numbered items (need {need}) {p['url']}")
            if len(items) >= need:
                # out = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(items))
                if save_to.lower().endswith(".csv"):
                    rows = []
                    for i, it in enumerate(items):
                        name, _, val = it.rpartition(" - ")
                        name = name.replace('"', '""')
                        val = val.replace("$", "").replace(",", "")
                        rows.append(f'{i + 1},"{name}",{val}')
                    out = "rank,name,value\n" + "\n".join(rows)
                else:
                    out = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(items))
                safe_path(save_to).write_text(out, encoding="utf-8")
                return (
                    f"DONE. All {len(items)} items saved to {save_to}. "
                    f"Source: {p['url']} (page title: {p['title']}). "
                    f"Preview of first 5 only: {', '.join(items[:5])}. "
                    f"Tell the user the file is complete with {len(items)} items."
                )
        return (f"error: no ranked page had a clean numbered list of {need}+ items. Tried:\n"
        + "\n".join(p["url"] for p in pages))

    best = pages[0]
    order = np.argsort(-best["chunk_scores"])[:2]
    parts = [f"[1] {best['chunks'][i][:1200]}" for i in order]
    for n, p in enumerate(pages[1:3], start=2):
        j = int(np.argmax(p["chunk_scores"]))
        parts.append(f"[{n + 1}] (from {p['url']}) {p['chunks'][j][:800]}")
    body = "\n\n---\n\n".join(parts)
    others = ", ".join(p["url"] for p in pages[1:3])
    items = extract_items(best["text"])
    if len(items) >= 5:
        body += "\n\nNumbered items on this page: " + "; ".join(items[:50])[:1500]
    return f"Best source: {best['url']}\nOther sources: {others}\n\n{body}"

# Local initialised tools
LOCAL_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "save_memory": save_memory,
    "run_command": run_command,
    "edit_file": edit_file,
    "web_search": web_search
}

# Tools Schema
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List all file names in the workspace directory. "
                "Call this first when you don't know which files exist, "
                "or before read_file to get the exact filename. "
                "Returns one filename per line, or '(empty)' if workspace has no files."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read full text content of one file in the workspace. "
                "Use exact filename as returned by list_files, including extension. "
                "Text files only (UTF-8). Returns file content, "
                "or a string starting with 'error:' if file does not exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Exact name of file to read, e.g. 'notes.txt'. No directory paths.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a new file in the workspace, or overwrite an existing one. "
                "Writes full contents as UTF-8 text. Existing content is replaced, not appended, "
                "so call read_file first if you need to keep or edit old content. "
                "Use for saving notes, summaries, code, or any text output the user wants stored. "
                "Returns 'wrote: <filename>' on success."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of file to write, e.g. 'summary.md'. Include extension. No directory paths.",
                    },
                    "contents": {
                        "type": "string",
                        "description": "Full text to write into file. Replaces any existing content.",
                    },
                },
                "required": ["filename", "contents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Save one lasting fact about the user to persistent memory, appended to memory.md. "
                "Use for stable info: name, job, preferences, goals, ongoing projects. "
                "Do NOT save temporary info, small talk, or facts already in memory. "
                "Write fact as one short self-contained sentence, e.g. 'User prefers Python over JavaScript.' "
                "Returns 'saved: <fact>' on success."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "One short sentence stating the fact about the user.",
                    }
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run one shell command on user's machine and return combined stdout and stderr. "
                "User must approve each command by typing 'y'; if declined, returns a refusal message. "
                "Use only when file tools cannot do job, e.g. run script, install package, check git status. "
                "Timeout 120 seconds. Returns output, or '(no output, exit code N)' if empty."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Full shell command to run, e.g. 'python script.py'. One command, no interactive prompts.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Quick web search, about 2 seconds. Returns 5 results: title, URL, snippet. "
                "Use for single current facts: office holders, prices, dates, scores, latest versions. "
                "Answer from the snippets when they contain the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Short search query, 2-8 words."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Change one exact piece of text in an existing workspace file. "
                "Use instead of write_file for edits. read_file first and copy old text exactly. "
                "old must appear exactly once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Exact file name, e.g. 'app.py'."},
                    "old": {"type": "string", "description": "Exact existing text to replace."},
                    "new": {"type": "string", "description": "Replacement text."},
                },
                "required": ["filename", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research",
            "description": (
                "Deep web research. Slow, 30-90 seconds. Opens 5 pages, ranks them, returns best passages and source URL. "
                "Use only when web_search snippets lack the answer, or for detail, comparison, or lists. "
                "Set save_to only to save a numbered list to a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Specific search question, 3-10 words."},
                    "save_to": {"type": "string", "description": "Optional filename, e.g. 'list.txt'. Only for saving numbered lists."},
                },
                "required": ["query"],
            },
        },
    },
]

# System insturctions for the AI
SYSTEM_PROMPT = """
You are a general assistant agent. You have a workspace folder, a shell, and web access. Today is {today}. Your training data is older than this.

## Tools
- list_files, read_file, write_file, edit_file: workspace files only. Flat names, no paths.
- run_command: shell in the workspace. User approves each call.
- web_search: quick search, about 2 seconds. Titles, URLs, snippets.
- research: deep web research, 30-90 seconds. Returns best passages and source URL. Set save_to only to save a numbered list to a file.
- get_current_time: current time.
- save_memory: only when the user asks you to remember something about them.

## Workflow
For every request, take the cheapest path that works.
1. Triage. Decide what the request needs.
   - Explanation, math, writing, or code from your own knowledge: answer directly. No tools.
   - Single current fact (office holder, price, date, score, latest version): web_search once. Answer from snippets.
   - Detail, comparison, list, or snippets lack the answer: research.
   - Work on files or code in the workspace: list_files, read_file, then change.
2. Investigate. Gather only what the answer needs. One tool call at a time. Read each result before the next call. The same call gives the same result, so change approach.
3. Act. Make the smallest change that solves it. New file: write_file. Change to existing file: edit_file.
4. Verify. Code: run it with run_command and read the output. Facts: the tool result must contain the answer. No evidence means say so.
5. Report. Answer first, then the source URL or what you ran. Short.

## Rules
- Never invent file contents, command output, URLs, names, or numbers. Use only what tools returned.
- Never write placeholders like [Name 6] or "..." in place of real content. Report the real count instead.
- Never answer current facts from memory. Search.
- Search or research returns an error or no answer: say so.
- A number can mean different things (city vs metro, 2011 vs 2026). State what it measures.
- Command fails: read the error, fix the cause, retry. Max 3 tries on the same failure, then stop and explain.
- No destructive commands (rm -rf, force push, drop table) without explicit user ask. Declined command: do not retry it.
- No commands that need interactive input or run forever.
- No secrets in files or memory.
- Tool result says "DONE. All N items saved": report N and the file name. The file is complete. A preview is only a preview, never call the list partial.
- Say what the source page ranks by (hottest, most popular, awards) when it differs from the user's request.
- Do not ask the user to choose between options. Pick the best, do it, report.
- Prefer simple, readable code. No extra features, no unrequested refactors.
- One question per search. Never pack several facts or several subjects into one query. For each subject and each fact, run its own query.
- Answer with the exact figure, name, or detail the user asked for. Snippets too vague to give it: call research. Never fill the gap with vague wording or memory.
- Check scope before using a source. It must cover the same subject, version, region, and date as the question. A source about a related or nearby subject is not evidence for this one. Name what the source covers. No match found: say "not found".
- A comparison needs the same measure for every subject. Missing for one subject: write "not found" for it and draw no conclusion about the difference. Never swap in a related measure.
- Say a file was saved only if a tool result says "wrote" or "DONE". No such result means no file exists.
- Python packages: use uv only. Never pip. Run a script that needs a package: `uv run --with NAME python file.py`. Add to the project permanently only if the user asks: `uv add NAME`.
- Research failed or no tool returned the data: say what failed and stop. Never fill a data file from memory.
- Saved item count must match the count the user asked for. If it differs, say so and fix it. Never excuse a mismatch.
- After write_file returns "wrote:", the file is saved. Do not write it again. If the user asked to run it, call run_command next.
- File request: do not give a final answer until write_file returns "wrote:" or research returns "DONE". If the data cannot be saved, say what failed. Never list partial or placeholder rows in chat.
- Top-N list to save as a file: call research with save_to. For .csv, research writes columns rank,name,value.
- Sum, count, or average of file data: run a command that reads the file. Never compute in your head.
- Shell is {shell}. On Windows never use awk, sed, cat, grep, bc, or bash loops.
- Math or file processing: write_file a small .py script with real newlines, then run_command `python name.py`. Never use `python -c` with for, with, or if.

## Style
Direct. No filler, no flattery. Code in fenced blocks. Disagree with the user when they are wrong, with reason.

## Memory
What you remember about the user:
{memory}
"""

mcp_session: dict[str, ClientSession] = {}

ALLOWED = {"get_current_time", "make_request"}
HIDDEN_TOOLS = {"make_request"}

# Make connection to MCP servers that are allowed
async def connect_mcp(stack: AsyncExitStack):
    for name, params in MCP_SERVERS.items():
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools = (await session.list_tools()).tools
        except Exception as e:
            print(f" [mcp] FAILED: {name}: {type(e).__name__}: {e}")
            continue

        loaded = []
        for tool in tools:
            if tool.name not in ALLOWED:
                continue
            mcp_session[tool.name] = session
            loaded.append(tool.name)
            if tool.name in HIDDEN_TOOLS:
                continue
            schema = getattr(tool, "input_schema", None) or getattr(tool, "input_schema", None) \
                or {"type": "object", "properties": {}}
            TOOL_SCHEMAS.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": schema,
                    },
                }
            )
        print(f" [mcp] connected: {name}: {loaded}")

# Call tools mcp or local as nesessary
async def call_tool(name: str, args: dict) -> str:
    if name == "research":
        return await research(**args)
    if name in LOCAL_TOOLS:
        return LOCAL_TOOLS[name](**args)
    result = await mcp_session[name].call_tool(name, args)
    return "\n".join(c.text for c in result.content if hasattr(c, "text"))[:5000]

MAX_STEPS = 20
RESULT_CAP = 3500

def trim_history(messages: list, max_chars: int = 16000) -> None:
    if sum(len(str(m)) for m in messages) <= max_chars:
        return
    tool_idx = [i for i, m in enumerate(messages)
                if isinstance(m, dict) and m.get("role") == "tool"]
    for i in tool_idx[:-2]:
        c = messages[i]["content"]
        if len(c) > 300:
            messages[i]["content"] = c[:300] + " [old result trimmed]"

def drop_old_turns(messages: list, max_chars: int = 16000) -> None:
    while sum(len(str(m)) for m in messages) > max_chars:
        user_idx = [i for i, m in enumerate(messages)
                    if isinstance(m, dict) and m.get("role") == "user"]
        if len(user_idx) < 3:
            return
        del messages[user_idx[0]:user_idx[1]]

def missing_args(name: str, args: dict) -> list[str]:
    for t in TOOL_SCHEMAS:
        f = t["function"]
        if f["name"] == name:
            return [k for k in f["parameters"].get("required", []) if k not in args]
    return []

# Agent harness loop
async def run_agent(messages: list) -> str:
    nudged = False
    searched = False
    seen = set()
    cmd_fails = 0
    for _ in range(MAX_STEPS):
        trim_history(messages=messages)
        drop_old_turns(messages=messages)
        t = time.time()
        try:
            response = await client.chat.completions.create(
                model= MODEL,
                messages= messages,
                tools= TOOL_SCHEMAS,
                temperature= 0.1,
                reasoning_effort = "none"
            )
        except Exception as e:
            return f"error: model call failed: {e}"
    
        print(f" [llm] {time.time() - t:.1f}s")
        message = response.choices[0].message
    
        if not message.tool_calls:
            text = message.content or ""
            if not text.strip() and not nudged:
                nudged = True
                messages.append({"role": "user", "content": "Answer now, briefly, from the tool results above."})
                continue
            if not text.strip():
                text = "(model returned an empty reply)"
            messages.append({"role": "assistant", "content": text})
            return text
    
        messages.append(message)
    
        for call in message.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
                print(f" [tool] {name}({args})")
                key = (name, json.dumps(args, sort_keys=True))
                if name == "research" and not searched and not args.get("save_to"):
                    result = "error: call web_search first. Use research only if snippets lack the answer."
                elif name == "run_command" and cmd_fails >= 5:
                    result = "error: 3 commands failed. Stop. Tell the user what failed and what you tried."
                elif key in seen and name != "run_command":
                    result = "error: identical call already made. Change the query or approach, or answer now."
                elif (miss := missing_args(name, args)):
                    result = f"error: {name} missing required args: {', '.join(miss)}. Resend the full call with all args."
                else:
                    seen.add(key)
                    if name == "web_search":
                        searched = True
                    result = str(await call_tool(name, args))[:RESULT_CAP]
                    if name == "write_file" and result.startswith("wrote"):
                        cmd_fails = 0
                    if name == "run_command" and any(s in result for s in ("is not recognized", "SyntaxError", "Traceback", "was unexpected")):
                        cmd_fails += 1
                print(f" [result] {result[:300]}")
            except Exception as e:
                traceback.print_exc()
                result = f"error: {type(e).__name__}: {e}"
                print(f" [error] {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "stopped: too many tool steps. Ask again with a narrower request."

# Main function to run it all
async def main():
    async with AsyncExitStack() as stack:
        await connect_mcp(stack)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(memory=load_memory(), today=date.today().isoformat(),  shell="Windows cmd.exe" if os.name == "nt" else "bash")}
        ]
        print(f"\nOffline Code")
        try:
            while True:
                messages.append({"role": "user", "content": input("\nyou: ")})
                print(f"\nAssistant: {await run_agent(messages)}")
        except (EOFError, KeyboardInterrupt):
                print("\nbye")


if __name__ == "__main__":
    asyncio.run(main())