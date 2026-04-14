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
                "You are a strict faithfulness judge for a RAG system.\n\n"
                "TASK: Decide if the Answer introduces ANY claim, fact, number, name, "
                "or term NOT explicitly present in the Context.\n\n"
                "STEP 1 — List every distinct factual claim in the Answer.\n"
                "STEP 2 — For each claim, write 'SUPPORTED' if it appears in the Context "
                "or 'NOT SUPPORTED' if it does not.\n"
                "STEP 3 — If every claim is SUPPORTED → final verdict: faithful.\n"
                "          If even ONE claim is NOT SUPPORTED → final verdict: unfaithful.\n\n"
                "End your response with exactly one line:\n"
                "VERDICT: faithful\n"
                "or\n"
                "VERDICT: unfaithful"
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
