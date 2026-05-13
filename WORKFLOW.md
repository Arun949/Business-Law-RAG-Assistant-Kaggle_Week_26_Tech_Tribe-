# Project Workflow — Business Law RAG Assistant

GenAI Hackathon · Week 26 | Tech Tribe

---

## Overview

This project is a **Retrieval-Augmented Generation (RAG)** system that answers Business Law questions from a private PDF corpus. It has two UI modes:

| Mode | Entry point | Pipeline |
|------|------------|----------|
| Single-agent (main) | `app/main.py` | Hybrid retrieval → GPT-4o-mini → LLM-as-Judge |
| Multi-agent | `app/gradio_app.py` | Agent 1 + Agent 2 + Agent 3 (Synthesizer) |

---

## End-to-End Pipeline

```
PDFs in data/sources/
        │
        ▼ (Step 1 — run once)
  scripts/extract.py
        │  PyMuPDF page-by-page extraction + text cleaning
        ▼
  data/corpus.json          ← {source, page, char_count, text} per page
        │
        ▼ (Step 2 — run once)
  app/build_index.py
        │  sentence-boundary chunking (800 chars, 120-char overlap)
        │  OpenAI text-embedding-3-small embeddings (batches of 100)
        │  L2 normalization → FAISS IndexFlatIP (cosine similarity)
        ▼
  data/my_index.faiss       ← vector index (loaded at query time)
  data/chunks.json          ← [{text, source, page}] for all 466 chunks
        │
        ▼ (Step 3 — runtime per query)
  User question (Gradio UI)
        │
        ├─► Query Reformulation (multi-turn)
        │       reformulate_query() rewrites the question into a
        │       standalone query by resolving pronouns via chat history
        │
        ▼
  Hybrid Retrieval (hybrid_retrieve)
        │  FAISS semantic search   → top-20 candidates
        │  BM25 keyword search     → top-20 candidates
        │  Reciprocal Rank Fusion  → top-5 chunks (RRF_K=60)
        │  Low-confidence warning if best score < 0.025
        ▼
  Context assembly
        │  [Excerpt N | source | page X] + chunk text
        │  Yes/No dynamic prompt injection
        ▼
  GPT-4o-mini (gpt-4o-mini)
        │  system prompt from prompts/v1.txt
        │  conversation history (last 4 turns)
        │  max_tokens=700, temperature=0
        ▼
  Answer
        │
        ├─► Refusal check (REFUSAL_PHRASES list)
        │         └─ if refusal → skip judge → "N/A" label
        │
        ├─► LLM-as-Judge (llm_judge.py)
        │         judge_faithfulness(question, answer, context)
        │         3-step CoT: list claims → SUPPORTED/NOT SUPPORTED → VERDICT
        │         verdict: faithful ✅ or unfaithful ❌
        │
        ├─► Cost tracking (cost_tracker.py)
        │         persistent data/cost_tracker.json
        │         broken down by calling script
        │
        └─► Gradio UI update
                  chatbot, sources panel, judge panel, cost display
```

---

## Step 1 — PDF Extraction (`scripts/extract.py`)

**Input:** `data/sources/*.pdf`  
**Output:** `data/corpus.json`, `data/sample.json`

- Uses **PyMuPDF** (`fitz`) to extract raw text page-by-page
- Cleans whitespace (multi-spaces, 3+ newlines, lone newlines)
- Flags low-text pages (<50 chars) as `image/diagram`, `likely blank`, or `low text`
- Saves every page as `{source, page, char_count, text}`
- Result: 469 pages from 1 PDF

---

## Step 2 — Index Building (`app/build_index.py`)

**Input:** `data/corpus.json`  
**Output:** `data/my_index.faiss`, `data/chunks.json`

| Parameter | Value |
|-----------|-------|
| Chunk target size | 800 chars |
| Overlap | 120 chars |
| Min chunk size | 50 chars |
| Embedding model | `text-embedding-3-small` |
| Batch size | 100 chunks |
| Index type | `FAISS IndexFlatIP` (cosine via L2 norm) |

- Splits pages into sentence-boundary-aware chunks
- Carries overlap sentences into the next chunk for context continuity
- Embeds all chunks in batches with a 200ms pause between batches (rate limit safety)
- Normalises vectors (L2) before adding to FAISS index
- Final index: **466 chunks**, 1536-dimensional vectors

---

## Step 3 — Query Time (Single-Agent: `app/main.py`)

### 3a. Query Reformulation
If there is conversation history, `reformulate_query()` calls GPT-4o-mini to rewrite the user's message as a self-contained search query (resolves pronouns, fills missing context).

### 3b. Hybrid Retrieval
```
FAISS semantic search (top-20)
        +
BM25 keyword search (top-20)
        ↓
Reciprocal Rank Fusion
        ↓
Top-5 chunks returned
```

RRF score per chunk: `1/(60 + faiss_rank) + 1/(60 + bm25_rank)`

### 3c. Answer Generation
- System prompt loaded from `prompts/v1.txt` (versioned, editable without code changes)
- Yes/No questions get a dynamic suffix instructing the model to lead with Yes/No
- Context assembled as labeled excerpts: `[Excerpt N | source.pdf | page X]`
- GPT-4o-mini generates answer grounded only in provided excerpts

### 3d. Faithfulness Verification (LLM-as-Judge)
`app/llm_judge.py` runs a 3-step chain-of-thought evaluation:
1. List every distinct factual claim in the answer
2. Mark each claim as `SUPPORTED` or `NOT SUPPORTED` against the context
3. Final `VERDICT: faithful` or `VERDICT: unfaithful`

Refusal answers skip the judge entirely.

### 3e. Cost Tracking
Every API call (chat + embeddings) records token usage to `data/cost_tracker.json`:
- `total_usd` — cumulative across all runs
- `calls` — number of API calls
- `by_script` — cost breakdown per script (`main`, `evaluate`, `llm_judge`, `agents`)

---

## Multi-Agent Pipeline (`app/gradio_app.py` + `app/agents.py`)

An alternative UI using a **3-agent orchestrator**:

```
User query
    │
    ├─► Agent 1: Internal Researcher
    │       hybrid_retrieve(query) → top-5 corpus chunks
    │
    ├─► Agent 2: External Fact-Checker
    │       search_web_with_sources(query)
    │       DuckDuckGo Instant Answer API → Wikipedia fallback
    │       returns [{url, title, snippet}]
    │
    └─► Agent 3: Synthesizer (GPT-4o-mini)
            Combines top-1 corpus chunk + web results
            Output: PART 1 (verbatim corpus) · PART 2 (web addition) · PART 3 (sources)
                │
                └─► LLM-as-Judge (same faithfulness check)
```

Special case: if "report" is in the query, `mcp_server.create_markdown_report()` saves a `.md` file to disk.

---

## MCP Server (`app/mcp_server.py`)

A **FastMCP** server exposing two tools:

| Tool | Description |
|------|-------------|
| `search_web(query)` | DuckDuckGo Instant Answer → Wikipedia fallback |
| `create_markdown_report(content, filename)` | Saves a markdown report to disk |

`search_web_with_sources(query)` is a non-MCP helper used directly by Agent 2 to return structured `{url, title, snippet}` dicts.

---

## Evaluation (`app/evaluate.py`)

**Input:** `data/questions.json` (19 questions across 4 categories)  
**Output:** `data/eval_results.json`

| Category | Questions | Pass condition |
|----------|-----------|---------------|
| Factual | 11 | answered (not refused) |
| Cross-reference | 3 | answered (not refused) |
| Out-of-scope | 3 | correct refusal |
| Ambiguous | 2 | clarified or broad answer |

Each answered question also gets a faithfulness verdict from LLM-as-Judge.

**Result:** 19/19 (100%) — all 14 answered questions verified faithful.

---

## Data Files

| File | Description |
|------|-------------|
| `data/sources/*.pdf` | Raw source PDFs (Business Law, 469 pages) |
| `data/corpus.json` | Extracted text per page |
| `data/chunks.json` | Chunked text with source + page metadata |
| `data/my_index.faiss` | FAISS vector index |
| `data/questions.json` | 19 evaluation questions |
| `data/eval_results.json` | Latest evaluation run results |
| `data/cost_tracker.json` | Cumulative API cost log |
| `prompts/v1.txt` | Versioned system prompt |

---

## How to Run (Quick Reference)

```bash
# 1. Extract corpus (skip if corpus.json exists)
python scripts/extract.py

# 2. Build FAISS index (skip if my_index.faiss exists)
python app/build_index.py

# 3a. Launch single-agent Gradio UI
python app/main.py

# 3b. Launch multi-agent Gradio UI
python app/gradio_app.py

# 4. Run evaluation suite
python app/evaluate.py
```

All UIs available at `http://127.0.0.1:7860`

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
| Web search | DuckDuckGo API + Wikipedia API |
| MCP server | FastMCP |
| UI | Gradio (`gr.Blocks`) |
| Cost tracking | Custom persistent JSON tracker |
| Env management | `python-dotenv` + `uv` |
