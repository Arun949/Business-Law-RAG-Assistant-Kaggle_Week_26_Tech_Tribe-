"""
eval_runner.py — run all questions.json through the retrieval pipeline and produce a scorecard.

Usage (from project root):
    python scripts/eval_runner.py               # retrieval check only (fast, cheap)
    python scripts/eval_runner.py --full        # retrieval + generate answers (costs tokens)
    python scripts/eval_runner.py --fix-chunks  # add expected_chunks to questions.json from source/page fields

Results saved to data/eval_results.json. Compare runs to catch regressions.
"""

import argparse, json, os, re, sys, time
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
RESULTS_PATH   = "data/eval_results.json"
CANDIDATE_K    = 20
RRF_K          = 60
TOP_K          = 5

CHAT_MODEL = "gpt-4o-mini"
COSTS = {"input": 0.150, "output": 0.600, "embed": 0.020}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def _embed(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    vec  = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def hybrid_retrieve(query, index, chunks, bm25, k=TOP_K):
    q_vec = _embed(query)
    faiss_scores, faiss_ids = index.search(q_vec, CANDIDATE_K)
    faiss_rank = {int(idx): rank for rank, idx in enumerate(faiss_ids[0]) if idx >= 0}

    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_top    = np.argsort(bm25_scores)[::-1][:CANDIDATE_K]
    bm25_rank   = {int(idx): rank for rank, idx in enumerate(bm25_top)}

    all_ids = set(faiss_rank) | set(bm25_rank)
    rrf = {
        idx: 1.0 / (RRF_K + faiss_rank.get(idx, CANDIDATE_K))
            + 1.0 / (RRF_K + bm25_rank.get(idx, CANDIDATE_K))
        for idx in all_ids
    }
    top_ids = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:k]
    return [(chunks[i], rrf[i]) for i in top_ids]


def generate_answer(question, hits):
    prompt_file = "prompts/v1.txt"
    if os.path.exists(prompt_file):
        with open(prompt_file, encoding="utf-8") as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = "Answer the question using only the provided excerpts."

    context = "\n\n---\n\n".join(
        f"[Excerpt {i+1} | {c['source']} | page {c['page']}]\n{c['text']}"
        for i, (c, _) in enumerate(hits)
    )
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": f"Context:\n\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=500,
    )
    answer = r.choices[0].message.content or ""
    cost = (r.usage.prompt_tokens * COSTS["input"] + r.usage.completion_tokens * COSTS["output"]) / 1e6
    return answer, cost


# ── Fix: auto-populate expected_chunks ────────────────────────────────────────

def fix_expected_chunks(questions):
    changed = 0
    for q in questions:
        if "expected_chunks" not in q and q.get("source") and q.get("page"):
            q["expected_chunks"] = [{"source": q["source"], "page": q["page"]}]
            changed += 1
    print(f"  Added expected_chunks to {changed} questions.")
    return questions


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",        action="store_true", help="Generate answers (costs tokens)")
    parser.add_argument("--fix-chunks",  action="store_true", help="Add expected_chunks to questions.json")
    args = parser.parse_args()

    if not os.path.exists(INDEX_PATH):
        sys.exit(f"ERROR: {INDEX_PATH} not found. Run python app/build_index.py first.")

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    print(f"Loaded {len(chunks)} chunks, {len(questions)} questions.\n")

    if args.fix_chunks:
        questions = fix_expected_chunks(questions)
        with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
        print(f"Saved {QUESTIONS_PATH}\n")

    results      = []
    total_cost   = 0.0
    hits_total   = 0
    by_category  = {}

    for i, q in enumerate(questions, 1):
        question        = q["question"]
        expected_chunks = q.get("expected_chunks", [])
        category        = q.get("category", "unknown")

        print(f"[{i:02d}/{len(questions)}] {question[:70]}...")

        hits = hybrid_retrieve(question, index, chunks, bm25)
        retrieved_sources = {(c["source"], c["page"]) for c, _ in hits}

        def _matches(e):
            src = e.get("source")
            page = e.get("page")
            pages = page if isinstance(page, list) else [page]
            return any((src, p) in retrieved_sources for p in pages)

        found = [e for e in expected_chunks if _matches(e)]
        retrieval_hit = len(found) == len(expected_chunks) if expected_chunks else None
        if retrieval_hit:
            hits_total += 1

        best_score = hits[0][1] if hits else 0

        result = {
            "question":      question,
            "category":      category,
            "retrieval_hit": retrieval_hit,
            "best_rrf":      round(best_score, 5),
            "retrieved":     [{"source": c["source"], "page": c["page"]} for c, _ in hits],
            "expected":      expected_chunks,
            "missed":        [e for e in expected_chunks if not _matches(e)],
        }

        if args.full:
            answer, q_cost = generate_answer(question, hits)
            total_cost += q_cost
            result["answer"]    = answer
            result["gen_cost"]  = round(q_cost, 6)
            print(f"         answer: {answer[:80].replace(chr(10),' ')}...")
            time.sleep(0.3)  # gentle rate-limit pause

        status = "✓" if retrieval_hit else ("?" if retrieval_hit is None else "✗")
        print(f"         retrieval={status}  RRF={best_score:.4f}  category={category}")

        by_category.setdefault(category, {"total": 0, "hits": 0})
        by_category[category]["total"] += 1
        if retrieval_hit:
            by_category[category]["hits"] += 1

        results.append(result)

    # ── Scorecard ──────────────────────────────────────────────────────────────
    questions_with_expected = [q for q in questions if q.get("expected_chunks")]
    hit_rate = hits_total / len(questions_with_expected) if questions_with_expected else None

    print(f"\n{'='*60}")
    print(f"  EVAL SCORECARD")
    print(f"{'='*60}")
    print(f"  Questions evaluated : {len(questions)}")
    if hit_rate is not None:
        print(f"  Retrieval hit rate  : {hits_total}/{len(questions_with_expected)} = {hit_rate:.0%}")
    if args.full:
        print(f"  Total API cost      : ${total_cost:.4f}")
    print(f"\n  By category:")
    for cat, stat in sorted(by_category.items()):
        if stat["hits"] > 0 or stat["total"] > 0:
            pct = stat["hits"] / stat["total"] if stat["total"] else 0
            print(f"    {cat:<20} {stat['hits']}/{stat['total']} ({pct:.0%})")

    missed_questions = [r for r in results if r["retrieval_hit"] is False]
    if missed_questions:
        print(f"\n  FAILED retrievals ({len(missed_questions)}):")
        for r in missed_questions:
            for m in r["missed"]:
                print(f"    ✗ {r['question'][:55]:<55}  → missed {m['source']} p.{m['page']}")
    print(f"{'='*60}\n")

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "run_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hit_rate":     round(hit_rate, 4) if hit_rate is not None else None,
        "total_cost":   round(total_cost, 6),
        "by_category":  by_category,
        "questions":    results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
