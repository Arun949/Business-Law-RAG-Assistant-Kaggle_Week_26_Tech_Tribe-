import json, sys, os, time
import numpy as np
import faiss, tiktoken, gradio as gr
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from cost_tracker import track_cost
from llm_judge import judge_faithfulness

load_dotenv()
client = OpenAI()
enc    = tiktoken.encoding_for_model("gpt-4o-mini")


# ── Section 4: Error handling with retry 

def safe_chat(messages, max_retries=3, **kwargs):
    """Call chat completions with exponential-backoff retry on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(messages=messages, **kwargs)
        except RateLimitError:
            time.sleep(2 ** attempt)
        except APIError as e:
            print(f"API error: {e}")
            time.sleep(1)
    return None

EMBED_MODEL, CHAT_MODEL = "text-embedding-3-small", "gpt-4o-mini"
TOP_K, MAX_HISTORY      = 5, 4
COSTS = {"input": 0.150, "output": 0.600, "embed": 0.020}  # $/1M tokens

SYSTEM_PROMPT = """\
You are a document question-answering assistant. Your answers must be grounded EXCLUSIVELY in the provided excerpts below — no outside knowledge.

RULES:
1. AMBIGUITY CHECK (apply this first): Only ask for clarification if the question is so vague that the answer would fundamentally differ depending on the missing context AND the excerpts cannot resolve it (e.g. "What are the penalties?" without specifying for what offence). Do NOT ask for clarification when: (a) the excerpts provide a clear single answer, (b) the question uses "my" or "I" but refers to a general process (e.g. "How do I file a patent?"), or (c) the topic has one standard answer in the corpus. If in doubt, answer from the excerpts.
2. VERBATIM ONLY: Copy the exact sentence(s) from the excerpts word-for-word. Do NOT add any introductory phrase (e.g. do not write "According to the document..." or "The text states..."). Do NOT add any connecting words, commentary, transitions, or explanations of your own. Your answer body must contain ONLY text that appears character-for-character in the excerpts.
3. If the answer spans multiple excerpts, paste each relevant passage verbatim one after the other separated by a blank line. Zero added words between them.
4. Include ALL sentences from the excerpt that are needed to fully answer the question — do not truncate.
5. OUT-OF-SCOPE CHECK: If the excerpts do not directly address the question (they only mention the topic in passing or cover a different subject), respond only with — "I don't have information about this in the available documents."
6. If the information is NOT found in any excerpt: respond only with — "I don't have information about this in the available documents."
7. End every answer with: **Sources:** listing ONLY the actual source filename and page numbers from the excerpts you used (e.g. **Sources:** AA00007386_00001.pdf, pp. 28, 130). Never invent or guess page numbers — use only those shown in the [Excerpt | source | page X] labels above.
"""

if not os.path.exists("data/my_index.faiss"):
    sys.exit("ERROR: Run  python app/build_index.py  first.")

index = faiss.read_index("data/my_index.faiss")
with open("data/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Ready — {len(chunks)} chunks indexed.")


def retrieve(query, k=TOP_K):
    emb_resp = client.embeddings.create(model=EMBED_MODEL, input=query)
    track_cost(emb_resp, is_embedding=True, script="main")
    q = np.array(emb_resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)
    scores, ids = index.search(q, k)
    return [(chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]


def rag_answer(question, history, session_cost):
    if not question.strip():
        return history, f"**Session cost:** ${session_cost:.4f}  \n**Last query:** $0.0000", session_cost, "", ""

    hits    = retrieve(question)
    context = "\n\n---\n\n".join(
        f"[Excerpt {i+1} | {c['source']} | page {c['page']}]\n{c['text']}"
        for i, (c, _) in enumerate(hits)
    )

    # Gradio 6 passes history as list of {"role": ..., "content": ...} dicts
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in (history or [])[-MAX_HISTORY * 2:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"})

    r      = safe_chat(messages, model=CHAT_MODEL, temperature=0, max_tokens=700)
    if r is None:
        return history, f"**Session cost:** ${session_cost:.4f}  \n**Last query:** $0.0000", session_cost, ""
    track_cost(r, script="main")
    answer = r.choices[0].message.content or ""

    q_cost = (r.usage.prompt_tokens * COSTS["input"] + r.usage.completion_tokens * COSTS["output"]) / 1e6
    q_cost += len(enc.encode(question)) * COSTS["embed"] / 1e6
    new_cost = session_cost + q_cost

    seen, src_lines = set(), []
    for c, _ in hits:
        if (k := (c["source"], c["page"])) not in seen:
            seen.add(k); src_lines.append(f"**{c['source']}**, p. {c['page']}")

    best_score  = hits[0][1] if hits else 0
    sources_md  = "**Retrieved excerpts:**\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(src_lines)) + f"\n\n*Best score: {best_score:.3f}*"
    cost_md     = f"**Session cost:** ${new_cost:.4f}  \n**Last query:** ${q_cost:.4f}"

    # ── LLM-as-Judge (live faithfulness check) ─────────────────────────────────
    REFUSAL_PHRASES = ("i don't have information", "not in the available documents",
                       "cannot find", "no information", "outside the scope",
                       "not covered", "i don't know", "unable to find")
    is_refusal = any(p in answer.lower() for p in REFUSAL_PHRASES)

    if is_refusal:
        judge_md = "### ⬜ N/A — Refusal answer, no faithfulness check needed."
    else:
        verdict, reasoning = judge_faithfulness(question, answer, context)
        if verdict == "faithful":
            judge_md = f"### ✅ Faithful\n\n<details><summary>View reasoning</summary>\n\n{reasoning}\n\n</details>"
        else:
            judge_md = f"### ❌ Unfaithful\n\n<details><summary>View reasoning</summary>\n\n{reasoning}\n\n</details>"

    updated_history = (history or []) + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]
    return updated_history, cost_md, new_cost, sources_md, judge_md


# ── Gradio UI 

EXAMPLES = [
    "What is a contract?", "What is the difference between civil law and criminal law?",
    "How does common law differ from civil law?", "What is a tort?", "What is jurisdiction?",
    "What is the weather in Paris today?", "Tell me about the requirements.",
]

with gr.Blocks(title="Business Law RAG Assistant") as demo:
    gr.Markdown("# 📚 Business Law RAG Assistant\nAnswers grounded in the corpus with source citations.\n> *GPT-4o-mini · text-embedding-3-small · FAISS*")

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            chatbot  = gr.Chatbot(label="Conversation", height=480, buttons=["copy"])
            with gr.Row():
                msg_box  = gr.Textbox(placeholder="Ask a question about business law…", label="", container=False, scale=5)
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)
            clear_btn = gr.Button("🗑  Clear conversation", variant="secondary", size="sm")
            gr.Examples(examples=EXAMPLES, inputs=msg_box, label="Example questions")

        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### 💰 Cost tracker")
            cost_display    = gr.Markdown("**Session cost:** $0.0000  \n**Last query:** $0.0000")
            gr.Markdown("---\n### 📄 Retrieved sources")
            sources_display = gr.Markdown("*Sources will appear here after your first query.*")
            gr.Markdown("---\n### 🔍 LLM-as-Judge")
            judge_display   = gr.Markdown("*Faithfulness verdict will appear here after your first query.*")
            gr.Markdown("---\n**Model:** gpt-4o-mini  \n**Embeddings:** text-embedding-3-small  \n**Top-K:** 5 chunks  \n**Corpus:** Business Law (469 pages)")

    cost_state = gr.State(0.0)
    outputs    = [chatbot, cost_display, cost_state, sources_display, judge_display]

    msg_box.submit(rag_answer, [msg_box, chatbot, cost_state], outputs).then(lambda: "", outputs=msg_box)
    send_btn.click(rag_answer, [msg_box, chatbot, cost_state], outputs).then(lambda: "", outputs=msg_box)
    clear_btn.click(
        lambda: ([], "**Session cost:** $0.0000  \n**Last query:** $0.0000", 0.0,
                 "*Sources will appear here after your first query.*",
                 "*Faithfulness verdict will appear here after your first query.*"),
        outputs=outputs
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft(primary_hue="blue"))
