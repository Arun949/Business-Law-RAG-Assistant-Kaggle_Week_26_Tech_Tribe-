# Business Law RAG Assistant

GenAI Hackathon — Week 26 | Tech Tribe

A Retrieval-Augmented Generation (RAG) application that answers natural language questions about Business Law documents with exact source citations, live faithfulness verification, and persistent cost tracking. Supports a single-agent Gradio UI, a 3-agent orchestrator, a Signal messenger bot, and an MCP server.

**Eval score: 19/19 (100%) · All 14 answered questions verified faithful by LLM-as-Judge**

---

## Team Members

1. Arun Kumar Aluru
2. Aishwarya Murthy
3. Sathwika Raj Bandaru
4. Nimisha Busaniwar

---

## What It Does

- Accepts questions via Gradio web UI, Signal messenger, or CLI
- Retrieves relevant passages using **hybrid search**: FAISS semantic + BM25 keyword with Reciprocal Rank Fusion (RRF)
- Rewrites follow-up questions to be self-contained using multi-turn conversation memory
- Generates verbatim-grounded answers via `gpt-4o-mini` (temperature=0) with exact page citations
- Refuses out-of-scope questions and asks clarification on ambiguous queries
- Verifies every answer with an **LLM-as-Judge** using 3-step chain-of-thought reasoning
- Tracks cumulative API costs per script in `data/cost_tracker.json` (budget: $5/week)

**Models:** `gpt-4o-mini` (chat) · `text-embedding-3-small` (embeddings)  
**Vector store:** FAISS `IndexFlatIP` (cosine similarity, 1536-dim, 466 chunks)  
**Keyword search:** BM25Okapi with RRF fusion (K=60)

---

## Architecture

### Single-Agent Pipeline (`app/main.py`)

```
User Question (Gradio)
    ↓
[Query Reformulation]       — resolves pronouns/context from last 4 turns
    ↓
[Hybrid Retrieval]
    ├─ FAISS semantic search (top-20)
    ├─ BM25 keyword search (top-20)
    └─ RRF fusion → top-5 chunks
    ↓
[GPT-4o-mini Answer Generation]
    ├─ System prompt from prompts/v1.txt
    ├─ Verbatim-only constraint + yes/no detection
    └─ max_tokens=700, temperature=0
    ↓
[LLM-as-Judge Faithfulness Check]
    ├─ Step 1: list every factual claim
    ├─ Step 2: mark SUPPORTED / NOT SUPPORTED
    └─ Step 3: VERDICT faithful ✅ or unfaithful ❌
    ↓
[Cost Tracking] → data/cost_tracker.json
    ↓
[Gradio UI] — chatbot + sources panel + judge verdict + session cost
```

### 3-Agent Orchestrator Pipeline (`app/gradio_app.py`)

| Agent | Role |
|-------|------|
| **Agent 1 — Internal Researcher** | Calls `hybrid_retrieve()` → top-5 corpus chunks |
| **Agent 2 — External Fact-Checker** | Calls DuckDuckGo + Wikipedia for live web results |
| **Agent 3 — Synthesizer** | Merges corpus excerpts + web findings; strict verbatim only |

The multi-agent UI adds **Human-in-the-Loop (HITL)** buttons:
- **Approve** → generates a timestamped `.md` report via MCP server
- **Add to DB** → embeds web results and appends them to the FAISS index (no full rebuild)
- **More Research** → re-runs Agent 1 with adjusted retrieval parameters

---

## Project Structure

```
.
├── app/
│   ├── main.py              # Single-agent Gradio UI + full RAG pipeline
│   ├── gradio_app.py        # 3-agent orchestrator UI with HITL buttons
│   ├── agents.py            # Agent 1 (Researcher), 2 (Fact-Checker), 3 (Synthesizer)
│   ├── build_index.py       # One-time: corpus → chunks → embeddings → FAISS index
│   ├── evaluate.py          # Batch evaluation: 19 questions → eval_results.json
│   ├── llm_judge.py         # LLM-as-Judge faithfulness verifier (3-step CoT)
│   ├── cost_tracker.py      # Persistent API cost logger (shared across scripts)
│   ├── mcp_server.py        # FastMCP server: web search + report generation + incremental indexing
│   ├── signal_bot.py        # Signal messenger bot (requires Docker + signal-cli-rest-api)
│   ├── launcher.py          # Starts Gradio + Signal bot in parallel
│   ├── run.py               # Simple CLI runner
│   ├── my_index.faiss       # Prebuilt FAISS index (466 chunks, 1536-dim)
│   └── chunks.json          # Chunk metadata (source, page, text)
├── scripts/
│   ├── extract.py           # PDF → corpus.json (PyMuPDF, page-by-page)
│   ├── eval_runner.py       # Retrieval-focused evaluation with --retrieval / --full modes
│   ├── retrieval_checker.py # Interactive CLI to debug retrieval failures
│   └── FAILURE_LOG.md       # Documented failure patterns and regression tracker
├── data/
│   ├── corpus.json          # 469 pages of extracted text (607K chars total)
│   ├── chunks.json          # 466 chunks with source + page metadata
│   ├── my_index.faiss       # FAISS vector index
│   ├── questions.json       # 19 evaluation questions across 4 categories
│   ├── eval_results.json    # Latest evaluation results with faithfulness verdicts
│   ├── cost_tracker.json    # Cumulative API cost log
│   └── sources/             # Input PDFs directory
├── prompts/
│   └── v1.txt               # Versioned system prompt (edit without touching code)
├── pyproject.toml           # Project config and dependencies (managed with uv)
├── WORKFLOW.md              # Detailed technical architecture and design decisions
└── .env                     # API keys (never committed)
```

---

## Installation

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
# 1. Clone the repo
git clone <repo-url>
cd "Kaggle_Week_26_Tech_Tribe 2"

# 2. Create virtual environment and install dependencies
uv venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install -r pyproject.toml

# 3. Add your OpenAI API key
echo "OPENAI_API_KEY=sk-proj-..." > .env
```

---

## How to Run

### Step 1 — Extract corpus from PDFs *(skip if `data/corpus.json` exists)*

Place PDFs in `data/sources/`, then:

```bash
python scripts/extract.py
```

Produces `data/corpus.json` (469 pages, 607K chars) and `data/sample.json` (first 10 pages preview).

### Step 2 — Build the FAISS index *(skip if `app/my_index.faiss` exists)*

```bash
python app/build_index.py
```

Chunks the corpus (800-char target, 120-char overlap), embeds with `text-embedding-3-small`, and saves `app/my_index.faiss` + `app/chunks.json`. Run **once** — loading from disk is free (~$0.009).

### Step 3 — Launch the web app

**Single-agent UI (recommended):**
```bash
python app/main.py
```

**3-agent orchestrator UI (with web search + HITL):**
```bash
python app/gradio_app.py
```

Both open at `http://127.0.0.1:7860`.

**Signal messenger bot** (requires Docker + signal-cli-rest-api):
```bash
python app/signal_bot.py
# or launch Gradio + Signal bot together:
python app/launcher.py
```

### Step 4 — Run the evaluation suite

```bash
python app/evaluate.py
```

Runs all 19 questions, prints a category-level report, saves verdicts to `data/eval_results.json`.

**Retrieval-focused evaluation:**
```bash
python scripts/eval_runner.py --retrieval   # fast, no LLM calls
python scripts/eval_runner.py --full        # full pipeline with costs
```

**Debug a specific query:**
```bash
python scripts/retrieval_checker.py "What is the definition of a contract?"
```

---

## Evaluation Results

| Category | Questions | Score |
|---|---|---|
| Factual | 11 | 11/11 (100%) |
| Cross-reference | 3 | 3/3 (100%) |
| Out-of-scope | 3 | 3/3 (100%) |
| Ambiguous | 2 | 2/2 (100%) |
| **Overall** | **19** | **19/19 (100%)** |

All 14 answered questions verified **faithful** by LLM-as-Judge (chain-of-thought, 3-step evaluation).

---

## Retrieval Design

| Component | Choice | Reason |
|---|---|---|
| Semantic search | FAISS `IndexFlatIP` | Cosine similarity over L2-normalized vectors |
| Keyword search | BM25Okapi | Catches exact legal terms missed by embeddings |
| Fusion | Reciprocal Rank Fusion (K=60) | Combines both rankings without score normalization |
| Chunk size | 800 chars target, 120 overlap | Balances context completeness with precision |
| Top-K | 20 per retriever → top-5 after RRF | Wide candidate pool, narrow final context |

---

## MCP Server Tools

`app/mcp_server.py` exposes three tools via FastMCP:

| Tool | Description |
|---|---|
| `search_web(query)` | DuckDuckGo Instant Answer + Wikipedia fallback |
| `create_markdown_report(content, filename)` | Saves `.md` report to disk |
| `add_to_database(snippets, urls, titles)` | Embeds web results and appends to FAISS index |

---

## Cost Tracking

All API usage is logged to `data/cost_tracker.json` broken down by script:

```json
{
  "total_usd": 0.0072,
  "calls": 55,
  "by_script": {
    "main": 0.0066,
    "evaluate": 0.0006,
    "llm_judge": 0.0000
  }
}
```

| Model | Input | Output |
|---|---|---|
| `gpt-4o-mini` | $0.150 / 1M tokens | $0.600 / 1M tokens |
| `text-embedding-3-small` | $0.020 / 1M tokens | — |

Budget: **$5.00 per team per week** (current total: ~$0.007)

---

## Prompts

The system prompt lives in `prompts/v1.txt` — versioned and editable without touching Python code. Key rules enforced:

- **VERBATIM ONLY:** copy exact sentences from excerpts; no added commentary
- **OUT-OF-SCOPE CHECK:** refuse if excerpts don't directly address the question
- **AMBIGUITY CHECK:** ask clarification only when truly unresolvable
- **SOURCES:** cite actual source filename and page numbers only (never invent)
- **YES/NO:** answer starts with "Yes" or "No" when question demands it

---

## Debugging

Common failure patterns and their fixes are documented in [`scripts/FAILURE_LOG.md`](scripts/FAILURE_LOG.md):

| Pattern | Symptom | Fix |
|---|---|---|
| A | Vector search misses exact legal terms | Add BM25 hybrid search |
| B | Answer cut mid-sentence | Increase `OVERLAP_CHARS` |
| C | LLM refuses despite relevant chunks | Loosen prompt strictness |
| D | LLM adds unsupported claims | Strengthen VERBATIM ONLY rule |
| E | Out-of-scope not refused | Add confidence threshold (RRF score < 0.025) |
| F | Yes/No returns paragraph | Detect question type, enforce concise format |
