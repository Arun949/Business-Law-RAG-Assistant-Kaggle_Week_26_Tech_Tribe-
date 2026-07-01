# Business Law RAG Assistant

GenAI Hackathon — Week 26 | Tech Tribe

A Retrieval-Augmented Generation (RAG) application that answers questions about Business Law documents with source citations and live faithfulness verification. Ships two UIs (single-agent and multi-agent), a hybrid FAISS+BM25 retriever, an MCP tool server for live web search, a Signal messenger bot, and an analytics dashboard.

---

## Team Members

1. Arun Kumar Aluru
2. Aishwarya Murthy
3. Sathwika Raj Bandaru
4. Nimisha Busaniwar

---

## What It Does

- Accepts natural language questions via a Gradio web UI (or Signal messenger)
- Retrieves relevant passages with **hybrid retrieval**: FAISS semantic search + BM25 keyword search, fused with Reciprocal Rank Fusion (RRF)
- Generates grounded answers using GPT-4o-mini with exact source and page citations
- Refuses out-of-scope questions gracefully and asks for clarification on ambiguous ones
- Verifies every answer live using an LLM-as-Judge faithfulness check
- Rewrites follow-up questions into standalone queries using conversation history (multi-turn memory)
- Optional **multi-agent mode**: an internal-researcher agent (corpus) + external-fact-checker agent (live web via MCP/DuckDuckGo/Wikipedia) + synthesizer agent, with human-in-the-loop Approve / Rewrite / Rephrase controls
- Human-in-the-loop approval can save an answer as a report and incrementally add fresh web results back into the FAISS index
- Tracks cumulative API cost per script in `data/cost_tracker.json`, visualized on an in-app Analytics dashboard

**Models:** `gpt-4o-mini` (chat) · `text-embedding-3-small` (embeddings)
**Vector store:** FAISS `IndexFlatIP` (cosine similarity) + BM25Okapi, fused via RRF
**Eval score:** 19/19 (100%) on `data/questions.json`

---

## Project Structure

```
.
├── app/
│   ├── main.py            # Single-agent Gradio UI — hybrid retrieval → GPT-4o-mini → LLM-as-Judge
│   ├── gradio_app.py       # Multi-agent Gradio UI — Agent 1 + Agent 2 + Agent 3 (Synthesizer) + HITL + Analytics tab
│   ├── agents.py           # 3-agent orchestrator (internal researcher, web fact-checker, synthesizer) + HITL helpers
│   ├── analytics.py        # Analytics dashboard tab (cost, eval hit-rate, faithfulness, RRF, feedback charts)
│   ├── mcp_server.py       # FastMCP server: search_web, create_markdown_report, incremental index updates
│   ├── signal_bot.py       # Signal messenger bot — routes messages through the agent pipeline
│   ├── launcher.py         # Starts Gradio + (optionally) the Signal bot as supervised subprocesses
│   ├── run.py              # Minimal terminal REPL for agents.agent()
│   ├── build_index.py      # One-time: chunk → embed → FAISS index
│   ├── evaluate.py         # Batch evaluation of app/main.py against questions.json (category pass/fail)
│   ├── cost_tracker.py     # Persistent cost tracker (shared across all scripts)
│   └── llm_judge.py        # LLM-as-Judge (faithfulness verification)
├── scripts/
│   ├── extract.py            # PDF → corpus.json extraction (PyMuPDF)
│   ├── eval_runner.py        # Retrieval-only (or full) eval scorecard against questions.json
│   ├── retrieval_checker.py  # CLI to inspect what hybrid_retrieve() returns for a single question
│   ├── EXTRACTION_REPORT.md  # Notes on corpus extraction quality (blank/image pages, etc.)
│   └── FAILURE_LOG.md
├── prompts/
│   └── v1.txt              # Versioned system prompt, editable without touching code
├── data/                    # NOT committed (gitignored) — see "Data Setup" below
└── WORKFLOW.md              # Detailed pipeline walkthrough
```

---

## Installation

**Requirements:** Python ≥3.11, [uv](https://github.com/astral-sh/uv)

```bash
# 1. Clone the repo
git clone <repo-url>
cd Kaggle_Week_26_Tech_Tribe

# 2. Create the environment and install all dependencies from pyproject.toml
uv sync
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env
```

`uv sync` installs everything declared in `pyproject.toml` (`openai`, `faiss-cpu`, `gradio`, `rank-bm25`, `fastmcp`, `PyMuPDF`, `signalbot`, `duckduckgo-search`, `matplotlib`, `pandas`, etc.) — no separate `pip install` list to keep in sync by hand.

---

## Data Setup

`data/` is **gitignored** and not part of this repository — that includes the source PDF, `corpus.json`, `chunks.json`, and the FAISS index. A fresh clone starts with an empty `data/` directory. To get a working app you need to either:

- **Regenerate from a PDF:** place your source PDF(s) in `data/sources/`, then run the extraction + indexing steps below, or
- **Copy an existing `data/` folder** (e.g. from a teammate) if you already have `corpus.json` / `my_index.faiss` / `chunks.json` generated.

`.env` (with your `OPENAI_API_KEY`) is also gitignored and must be created locally per the Installation step above.

---

## How to Run

### Step 1 — Extract corpus from PDFs (skip if `data/corpus.json` already exists)

Place your PDFs in `data/sources/`, then run:

```bash
python scripts/extract.py
```

Produces `data/corpus.json` and `data/sample.json`.

### Step 2 — Build the FAISS index (skip if `data/my_index.faiss` already exists)

```bash
python app/build_index.py
```

Chunks the corpus (sentence-boundary aware, 800 chars, 120-char overlap), embeds with `text-embedding-3-small`, saves `data/my_index.faiss` and `data/chunks.json`. Do this **once** — loading from disk afterward is free.

### Step 3a — Launch the single-agent Gradio UI

```bash
python app/main.py
```

Opens at `http://127.0.0.1:7860`. Every query:
- Reformulates follow-up questions using conversation history
- Retrieves top-5 chunks via hybrid FAISS + BM25 → RRF
- Generates an answer via GPT-4o-mini, grounded only in retrieved excerpts
- Runs LLM-as-Judge to verify faithfulness
- Logs cost to `data/cost_tracker.json`

### Step 3b — Launch the multi-agent Gradio UI

```bash
python app/gradio_app.py
```

A 3-agent pipeline (internal researcher → external web fact-checker → synthesizer) with a **💬 Chat** tab and a **📊 Analytics** tab (live cost, eval hit-rate, faithfulness, and RRF charts for the current session). Includes human-in-the-loop controls: Approve (save as report), Rewrite (redo with more corpus or fresh web results), Rephrase (full re-run).

### Step 3c — Launch everything via the supervised launcher (optional)

```bash
python app/launcher.py
```

Starts the multi-agent Gradio UI as a supervised subprocess, and also starts the Signal bot ([app/signal_bot.py](app/signal_bot.py)) if `SIGNAL_PHONE_NUMBER` is set in `.env` and a `signal-cli-rest-api` container is reachable on `localhost:8080`. Auto-restarts either process if it crashes after a healthy run; see the docstring in [app/signal_bot.py](app/signal_bot.py) for Signal setup instructions.

### Step 4 — Run the evaluation suite

```bash
python app/evaluate.py
```

Runs all questions from `data/questions.json` through the single-agent pipeline, prints a category-level report, and saves detailed results (including faithfulness verdicts) to `data/eval_results.json`.

For a cheaper, retrieval-only check (no generation cost), or to debug a single question:

```bash
python scripts/eval_runner.py                              # retrieval-only scorecard
python scripts/eval_runner.py --full                        # + answer generation
python scripts/retrieval_checker.py "What is a contract?"   # inspect retrieval for one question
```

> `app/evaluate.py` and `scripts/eval_runner.py` both write to `data/eval_results.json` using the same `hit_rate` / `by_category` schema — running either one is safe, but running one after the other overwrites the previous run's results file.

---

## Evaluation Results

| Category | Score |
|---|---|
| Factual | 11/11 (100%) |
| Cross-reference | 3/3 (100%) |
| Out-of-scope | 3/3 (100%) |
| Ambiguous | 2/2 (100%) |
| **Overall** | **19/19 (100%)** |

All answered questions verified **faithful** by LLM-as-Judge (chain-of-thought evaluation).

---

## Cost Tracking

All API costs are automatically logged to `data/cost_tracker.json`, broken down by calling script (`main`, `evaluate`, `llm_judge`, `agents`):

```json
{
  "total_usd": 0.128041,
  "calls": 681,
  "last_updated": "2026-04-17T00:44:25",
  "by_script": {
    "main": 0.044632,
    "evaluate": 0.001374,
    "llm_judge": 0.056251,
    "agents": 0.025784
  }
}
```

The **📊 Analytics** tab in `app/gradio_app.py` visualizes this file live, alongside evaluation hit-rate, session faithfulness verdicts, RRF score trends, and 👍/👎 user feedback (`data/feedback_log.jsonl`).

---

## MCP Server

`app/mcp_server.py` runs a [FastMCP](https://github.com/jlowin/fastmcp) server exposing:

| Tool | Description |
|------|-------------|
| `search_web(query)` | DuckDuckGo Instant Answer API → Wikipedia fallback |
| `create_markdown_report(content, filename)` | Saves a markdown report to disk |
| `add_to_database(snippets, urls, titles)` | Embeds and appends new web-sourced chunks into the live FAISS index |

`search_web_with_sources(query)` is a non-MCP helper (used directly by Agent 2) that returns structured `{url, title, snippet}` results, preferring `duckduckgo-search` with a Wikipedia fallback.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| PDF extraction | PyMuPDF (`fitz`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | FAISS `IndexFlatIP` |
| Keyword search | BM25Okapi (`rank_bm25`) |
| Rank fusion | Reciprocal Rank Fusion (RRF) |
| LLM | OpenAI `gpt-4o-mini` |
| Web search | DuckDuckGo (`duckduckgo-search`) + Wikipedia API |
| MCP server | FastMCP |
| UI | Gradio (`gr.Blocks`) |
| Messaging | Signal (`signalbot` + `signal-cli-rest-api`) |
| Analytics | Matplotlib + Pandas |
| Cost tracking | Custom persistent JSON tracker |
| Env / package management | `python-dotenv` + `uv` |

See [WORKFLOW.md](WORKFLOW.md) for a detailed step-by-step pipeline walkthrough.
