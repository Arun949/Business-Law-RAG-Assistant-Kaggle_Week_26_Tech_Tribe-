# RAG Failure Log

Track every failed question here. Update as you iterate.

---

## How to use this file

1. Run `python scripts/retrieval_checker.py "your question"` to diagnose a failure.
2. Identify whether the problem is **retrieval** (wrong chunks) or **generation** (right chunks, wrong answer).
3. Apply a fix and re-run `python app/evaluate.py` to verify it worked and nothing regressed.
4. Log the result below.

---

## Common failure patterns

### Pattern A — Vector search misses exact terms
**Symptom:** Score < 0.30, or the right chunk is not retrieved at all.  
**Diagnosis:** `retrieval_checker.py` shows the expected chunk is MISSED.  
**Cause:** The query uses exact article numbers, codes, or acronyms (e.g. "Article 12.3", "CISG") that embedding models handle poorly.  
**Fix:** Add BM25 hybrid search (`rank_bm25`) and combine scores with Reciprocal Rank Fusion.

---

### Pattern B — Right source, wrong page / chunk boundary splits the answer
**Symptom:** Score is OK (≥ 0.40) but the retrieved chunk contains only half the answer sentence.  
**Diagnosis:** `retrieval_checker.py` shows the source is correct but the page is off by 1, or the text preview cuts mid-sentence.  
**Cause:** `CHUNK_TARGET_CHARS` is too small, or `OVERLAP_CHARS` is too small to carry the sentence across chunk boundaries.  
**Fix:** Increase `OVERLAP_CHARS` in `app/build_index.py` (try 200), rebuild index.

---

### Pattern C — LLM refuses despite relevant chunks being retrieved
**Symptom:** Retrieval checker shows expected chunks ARE in top-k, but the app responds "I don't have information about this."  
**Diagnosis:** The retrieved text is in the context but phrased differently from the question; the LLM decides it's not a direct answer.  
**Cause:** Overly strict system prompt, or the relevant sentence is buried deep in a long chunk.  
**Fix:** Reduce `CHUNK_TARGET_CHARS` so chunks are more focused, or loosen the VERBATIM ONLY rule in the prompt for that question type.

---

### Pattern D — LLM gives a wrong answer despite correct retrieval
**Symptom:** LLM-as-Judge marks answer as UNFAITHFUL. Retrieval checker shows expected chunks present.  
**Diagnosis:** The model is paraphrasing instead of quoting, or hallucinating.  
**Fix:** Strengthen the VERBATIM ONLY rule in the system prompt. Add "Copy the exact sentence character-for-character" emphasis. Lower temperature (already 0 — check if it drifted).

---

### Pattern E — Out-of-scope question not refused
**Symptom:** The app gives an answer to a question clearly outside the corpus (e.g. "What is the weather in Paris?").  
**Diagnosis:** A chunk happened to be retrieved with a borderline score (~0.25) and the LLM treated it as relevant.  
**Fix:** Add a confidence threshold check: if `best_score < 0.35`, prepend the out-of-scope response regardless of the LLM output.

---

### Pattern F — Yes/no question gets a paragraph
**Symptom:** Question is "Is X allowed?" but answer is a long paragraph.  
**Diagnosis:** Generation problem — prompt doesn't instruct concise format for yes/no questions.  
**Fix:** Detect yes/no questions and add "Start your answer with Yes or No." to the user message.

---

## Failure log table

| # | Question | Category | Retrieval OK? | Problem | Fix applied | Fixed? |
|---|----------|----------|:---:|---------|-------------|:---:|
| 1 | "How many years does law school take in the United States, and what degree do graduates earn?"_ | Factual | Yes | Vector search missed exact article number | — | — |

> Add your own rows below. Mark Fixed? with ✓ or ✗.

---

## Regression tracker

After each fix, paste a one-line summary here so you can track trade-offs:

| Date | Change | Questions fixed | Regressions |
|------|--------|-----------------|-------------|
| — | — | — | — |
