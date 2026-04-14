import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from main import rag_answer                      # noqa: E402  (after sys.path edit)

# ── Configuration 
QUESTIONS_PATH = "data/questions.json"
RESULTS_PATH   = "data/eval_results.json"


REFUSE_PHRASES = [
    "i don't have information",
    "not in the available documents",
    "cannot find",
    "no information",
    "outside the scope",
    "not covered",
    "i don't know",
    "unable to find",
]

# ── Helpers 

def did_refuse(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in REFUSE_PHRASES)


def judge_factual(answer: str, expected: str | None) -> str:
    if did_refuse(answer):
        return "refused"
    return "answered"


def judge_out_of_scope(answer: str) -> str:
    return "correct_refusal" if did_refuse(answer) else "incorrect_answer"


def judge_ambiguous(answer: str) -> str:
    if did_refuse(answer):
        return "refused"
    if "?" in answer:
        return "clarified"
    return "broad_answer"


# ── Main 

def main():
    # Load questions
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions: list[dict] = json.load(f)

    print(f"Loaded {len(questions)} evaluation questions.\n")
    print("=" * 60)

    results:     list[dict] = []
    total_cost   = 0.0
    by_category: dict[str, dict] = {}

    for i, q in enumerate(questions, 1):
        category = q["category"]
        question = q["question"]
        expected = q.get("expected_answer")

        print(f"[{i:02d}/{len(questions)}] [{category.upper():14s}]  {question}")

        # Run through the RAG pipeline (no history, fresh turn)
        history, cost_md, updated_cost, sources_md, judge_md = rag_answer(
            question    = question,
            history     = [],
            session_cost = total_cost,
        )
        answer     = history[-1]["content"] if history else ""
        query_cost = updated_cost - total_cost
        total_cost = updated_cost

        # Judge
        faithfulness = None
        if category in ("factual", "cross-reference"):
            verdict = judge_factual(answer, expected)
            if verdict == "answered":
                # Extract faithfulness from judge_md (already computed by rag_answer with correct context)
                faithfulness = "faithful" if "✅" in judge_md else "unfaithful" if "❌" in judge_md else None
        elif category == "out-of-scope":
            verdict = judge_out_of_scope(answer)
        elif category == "ambiguous":
            verdict = judge_ambiguous(answer)
        else:
            verdict = "unknown"

        print(f" verdict : {verdict}")
        if faithfulness:
            print(f" faithfulness : {faithfulness}")
        print(f" cost : ${query_cost:.4f}")
        print()

        result = {
            "question":        question,
            "category":        category,
            "expected_answer": expected,
            "system_answer":   answer,
            "verdict":         verdict,
            "faithfulness":    faithfulness,
            "sources_info":    sources_md,
            "query_cost_usd":  round(query_cost, 6),
        }
        results.append(result)

        if category not in by_category:
            by_category[category] = {"total": 0, "pass": 0}
        by_category[category]["total"] += 1

        passed = (
            (category in ("factual", "cross-reference") and verdict == "answered")
            or (category == "out-of-scope"  and verdict == "correct_refusal")
            or (category == "ambiguous"     and verdict in ("clarified", "broad_answer"))
        )
        if passed:
            by_category[category]["pass"] += 1

        # Small delay to stay within rate limits
        time.sleep(0.3)

    # ── Summary report 
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    total_pass  = 0
    total_total = 0
    for cat, counts in by_category.items():
        pct = counts["pass"] / counts["total"] * 100 if counts["total"] else 0
        total_pass  += counts["pass"]
        total_total += counts["total"]
        print(f"  {cat:<20s}  {counts['pass']}/{counts['total']}  ({pct:.0f}%)")

    overall_pct = total_pass / total_total * 100 if total_total else 0
    print(f"\n  {'OVERALL':<20s}  {total_pass}/{total_total}  ({overall_pct:.0f}%)")
    print(f"\n  Total cost: ${total_cost:.4f}")
    print("=" * 60)

    # ── Save results 
    output = {
        "run_at":    datetime.now().isoformat(timespec="seconds"),
        "summary":   {
            cat: {
                "pass":  v["pass"],
                "total": v["total"],
                "pct":   round(v["pass"] / v["total"] * 100, 1) if v["total"] else 0,
            }
            for cat, v in by_category.items()
        },
        "overall": {
            "pass":       total_pass,
            "total":      total_total,
            "pct":        round(overall_pct, 1),
            "total_cost": round(total_cost, 6),
        },
        "results": results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅  Detailed results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
