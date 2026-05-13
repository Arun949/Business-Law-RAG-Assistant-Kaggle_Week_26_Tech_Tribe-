# Business Law RAG Assistant

GenAI Hackathon — Week 26 | Tech Tribe

A Retrieval-Augmented Generation (RAG) application that answers questions about Business Law documents with source citations and live faithfulness verification.

---

## Team Members

1. Arun Kumar Aluru
2. Aishwarya Murthy
3. Sathwika Raj Bandaru
4. Nimisha Busaniwar

---

## What It Does

- Accepts natural language questions via a Gradio web UI
- Retrieves the most relevant passages from a Business Law corpus (469 pages, 466 chunks)
- Generates grounded answers using GPT-4o-mini with exact source and page citations
- Refuses out-of-scope questions gracefully
- Asks for clarification on ambiguous questions
- Verifies every answer live using an LLM-as-Judge faithfulness check
- Tracks cumulative API cost per script in `data/cost_tracker.json`

**Models:** `gpt-4o-mini` (chat) · `text-embedding-3-small` (embeddings)  
**Vector store:** FAISS `IndexFlatIP` (cosine similarity)  
**Eval score:** 19/19 (100%) on `data/questions.json`

---

## Project Structure

```
.
├── app/
│   ├── main.py           # Gradio web app — RAG pipeline + LLM-as-Judge UI
│   ├── build_index.py    # One-time: chunk → embed → FAISS index
│   ├── evaluate.py       # Batch evaluation against questions.json
│   ├── cost_tracker.py   # Persistent cost tracker (shared across all scripts)
│   ├── llm_judge.py      # LLM-as-Judge (faithfulness verification)
├── scripts/
│   └── extract.py        # PDF → corpus.json extraction (PyMuPDF)
|   └── extraction_report
| .gitigonre    
```

---

## Installation

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv)

```bash
# 1. Clone the repo
git clone <repo-url>
cd Kaggle_Week_26_Tech_Tribe

# 2. Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install openai python-dotenv tiktoken faiss-cpu numpy gradio pymupdf

# 3. Add OpenAI API key
echo "..." > .env
```

---

## How to Run

### Step 1 — Extract corpus from PDFs (already done, skip if corpus.json exists)

Place your PDFs in `data/sources/`, then run:

```bash
python scripts/extract.py
```

Produces `data/corpus.json`.

### Step 2 — Build the FAISS index (already done, skip if my_index.faiss exists)

```bash
python app/build_index.py
```

Chunks the corpus, embeds with `text-embedding-3-small`, saves `app/my_index.faiss` and `app/chunks.json`. Do this **once** — loading from disk is free.

### Step 3 — Launch the web app

```bash
python app/main.py
```

Opens the Gradio UI at `http://127.0.0.1:7860`. Every query automatically:
- Retrieves top-5 chunks from FAISS
- Generates an answer via GPT-4o-mini
- Runs LLM-as-Judge to verify faithfulness
- Logs cost to `data/cost_tracker.json.`

### Step 4 — Run the evaluation suite

```bash
python app/evaluate.py
```

Runs all  questions from `data/questions.json` through the pipeline, prints a category-level report, and saves detailed results (including faithfulness verdicts) to `data/eval_results.json`.

---

## Evaluation Results

| Category | Score |
|---|---|
| Factual | 11/11 (100%) |
| Cross-reference | 3/3 (100%) |
| Out-of-scope | 3/3 (100%) |
| Ambiguous | 2/2 (100%) |
| **Overall** | **19/19 (100%)** |

All  answered questions verified **faithful** by LLM-as-Judge (chain-of-thought evaluation).

---

## Cost Tracking

All API costs are automatically logged to `data/cost_tracker.json` broken down by script:

```json
{
  "total": 0.001215,
  "total_usd": 0.068183,
  "calls": 342,
  "last_updated": "2026-04-14T14:06:21",
  "by_script": {
    "main": 0.038018,
    "evaluate": 0.001374,
    "llm_judge": 0.028791
  }
}
```

