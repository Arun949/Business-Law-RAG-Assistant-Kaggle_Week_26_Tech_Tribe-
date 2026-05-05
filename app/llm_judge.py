from openai import OpenAI
from dotenv import load_dotenv
from cost_tracker import track_cost

load_dotenv()
_client = OpenAI()

def judge_faithfulness(question: str, answer: str, context: str) -> tuple[str, str]:
    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a faithfulness judge for a RAG system.\n\n"
                "The Answer may contain sentences from a private corpus AND from the web. "
                "You are given only the corpus Context.\n\n"
                "TASK: Check only the sentences that appear verbatim or near-verbatim in the "
                "corpus Context. Sentences NOT found in the Context are web-sourced — ignore them.\n\n"
                "STEP 1 — For each sentence in the Answer:\n"
                "  - If it appears verbatim or near-verbatim in the Context → label it CORPUS.\n"
                "  - If it does NOT appear in the Context → label it WEB (skip, do not evaluate).\n"
                "STEP 2 — For each CORPUS sentence: write SUPPORTED.\n"
                "          If it misquotes or distorts the Context: write NOT SUPPORTED.\n"
                "STEP 3 — All CORPUS sentences SUPPORTED (or no CORPUS sentences found) → faithful.\n"
                "          Any CORPUS sentence NOT SUPPORTED → unfaithful.\n\n"
                "End with exactly one line:\n"
                "VERDICT: faithful\nor\nVERDICT: unfaithful"
            )},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Context:\n{context}\n\n"
                f"Answer:\n{answer}"
            )},
        ],
        temperature=0,
        max_tokens=800,
    )
    track_cost(response, script="llm_judge")
    reasoning = response.choices[0].message.content.strip()
    lower = reasoning.lower()

    verdict = "unfaithful"
    for line in reversed(lower.splitlines()):
        if "verdict:" in line:
            verdict = "unfaithful" if "unfaithful" in line else "faithful"
            break

    return verdict, reasoning
