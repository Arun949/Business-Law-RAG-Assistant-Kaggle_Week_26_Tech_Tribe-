"""
retrieval_checker.py — inspect what the RAG system actually retrieves for a question.

Uses the same hybrid (FAISS + BM25 → RRF) retrieval as the main app.

Usage (run from project root):
    python scripts/retrieval_checker.py "What is a contract?"
    python scripts/retrieval_checker.py "What does Article 12.3 say?" --k 8

Output:
    - Top-k retrieved chunks with source, page, and RRF score
    - Whether the retrieved sources match the expected sources from questions.json

Common failure patterns:
    Low RRF scores overall  → neither vector nor keyword search found a strong match.
                              Try rephrasing or check if the term appears in the corpus.
    Score OK but wrong chunk → chunk boundary cut the key sentence.
                               Increase OVERLAP_CHARS in build_index.py and rebuild.
    Right source, wrong page → answer spans pages; try --k 10 or larger OVERLAP_CHARS.
    Exact term missed        → check BM25 rank column; if BM25 also missed, the term
                               may not appear verbatim in the corpus.
"""

import argparse, json, os, re, sys
import numpy as np
import faiss
from openai import OpenAI
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

EMBED_MODEL    = "text-embedding-3-small"
INDEX_PATH     = "data/my_index.faiss"
CHUNKS_PATH    = "data/chunks.json"
QUESTIONS_PATH = "data/questions.json"
CANDIDATE_K    = 20
RRF_K          = 60


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def hybrid_retrieve(query: str, index, chunks, bm25, k: int):
    # FAISS
    resp  = client.embeddings.create(model=EMBED_MODEL, input=query)
    q_vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_vec)
    faiss_scores, faiss_ids = index.search(q_vec, CANDIDATE_K)
    faiss_rank = {int(idx): rank for rank, idx in enumerate(faiss_ids[0]) if idx >= 0}

    # BM25
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_top    = np.argsort(bm25_scores)[::-1][:CANDIDATE_K]
    bm25_rank   = {int(idx): rank for rank, idx in enumerate(bm25_top)}

    # RRF
    all_ids = set(faiss_rank) | set(bm25_rank)
    rrf = {
        idx: 1.0 / (RRF_K + faiss_rank.get(idx, CANDIDATE_K))
            + 1.0 / (RRF_K + bm25_rank.get(idx, CANDIDATE_K))
        for idx in all_ids
    }
    top_ids = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:k]
    return [(chunks[i], rrf[i], faiss_rank.get(i), bm25_rank.get(i)) for i in top_ids]


def load_expected(question: str) -> list[dict]:
    if not os.path.exists(QUESTIONS_PATH):
        return []
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        qs = json.load(f)
    q_lower = question.lower().strip()
    for q in qs:
        if q.get("question", "").lower().strip() == q_lower:
            return q.get("expected_chunks", [])
    return []


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG retrieval inspector.")
    parser.add_argument("question", help="Question to retrieve for")
    parser.add_argument("--k", type=int, default=5, help="Chunks to return (default: 5)")
    args = parser.parse_args()

    if not os.path.exists(INDEX_PATH):
        sys.exit(f"ERROR: {INDEX_PATH} not found. Run python app/build_index.py first.")

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])

    expected = load_expected(args.question)

    print(f"\n{'='*60}")
    print(f"  RETRIEVAL CHECKER  (hybrid FAISS + BM25 → RRF)")
    print(f"{'='*60}")
    print(f"  Question : {args.question}")
    print(f"  Top-K    : {args.k}  |  Candidates per retriever: {CANDIDATE_K}")
    print(f"{'='*60}\n")

    hits = hybrid_retrieve(args.question, index, chunks, bm25, args.k)
    retrieved_sources = {(c["source"], c["page"]) for c, *_ in hits}

    for rank, (chunk, rrf_score, f_rank, b_rank) in enumerate(hits, 1):
        match_marker = ""
        if expected:
            for exp in expected:
                if chunk["source"] == exp.get("source") and chunk["page"] == exp.get("page"):
                    match_marker = "  ✓ EXPECTED"
                    break
        f_str = f"faiss_rank={f_rank}" if f_rank is not None else "faiss_rank=—"
        b_str = f"bm25_rank={b_rank}"  if b_rank  is not None else "bm25_rank=—"
        print(f"--- Rank {rank} | RRF={rrf_score:.4f} | {f_str} | {b_str}{match_marker} ---")
        print(f"    Source : {chunk['source']}, page {chunk['page']}")
        preview = chunk["text"][:300].replace("\n", " ")
        print(f"    Text   : {preview}{'...' if len(chunk['text']) > 300 else ''}")
        print()

    # Expected chunk audit
    if expected:
        print(f"\n{'='*60}")
        print("  EXPECTED CHUNK AUDIT")
        print(f"{'='*60}")
        found  = [e for e in expected if (e.get("source"), e.get("page")) in retrieved_sources]
        missed = [e for e in expected if (e.get("source"), e.get("page")) not in retrieved_sources]
        print(f"  Found : {len(found)}/{len(expected)} expected chunks in top-{args.k}")
        if missed:
            print(f"\n  MISSED chunks:")
            for m in missed:
                print(f"    - {m.get('source')}, page {m.get('page')}")
            print("\n  Likely causes:")
            print("    1. Term doesn't appear verbatim → check corpus text for spelling variants")
            print("    2. Chunk boundary split the key sentence → increase OVERLAP_CHARS")
            print("    3. TOP_K too small → try --k 10")
            print("    4. Chunk too large → key text diluted → reduce CHUNK_TARGET_CHARS")
        print(f"{'='*60}\n")
    else:
        print("  (No expected chunks found in questions.json for this question)")


if __name__ == "__main__":
    main()
