import json, sys, os, time, re
import numpy as np
import faiss, tiktoken, gradio as gr
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from cost_tracker import track_cost
from llm_judge import judge_faithfulness

load_dotenv()
client = OpenAI()
enc    = tiktoken.encoding_for_model("gpt-4o-mini")


# ── Error handling with retry ──────────────────────────────────────────────────

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


# ── Constants ──────────────────────────────────────────────────────────────────

EMBED_MODEL, CHAT_MODEL = "text-embedding-3-small", "gpt-4o-mini"
TOP_K, MAX_HISTORY      = 5, 4
CANDIDATE_K             = 20     # candidates fetched from each retriever before RRF
RRF_K                   = 60     # RRF rank constant (standard value)
RRF_THRESHOLD           = 0.025  # below this both retrievers found weak matches → warn
COSTS = {"input": 0.150, "output": 0.600, "embed": 0.020}

FEEDBACK_LOG = "data/feedback_log.jsonl"


# ── Prompt versioning ──────────────────────────────────────────────────────────
# Edit prompts/v1.txt to change the system prompt without touching this file.
# Copy to prompts/v2.txt before making changes so you can roll back.

_PROMPT_FILE = "prompts/v1.txt"

def _load_system_prompt() -> str:
    if os.path.exists(_PROMPT_FILE):
        with open(_PROMPT_FILE, encoding="utf-8") as f:
            return f.read().strip()
    # fallback inline (should not happen if prompts/ exists)
    return """\
You are a document question-answering assistant. Your answers must be grounded EXCLUSIVELY in the provided excerpts below — no outside knowledge.

RULES:
1. AMBIGUITY CHECK (apply this first): Only ask for clarification if the question is so vague that the answer would fundamentally differ depending on the missing context AND the excerpts cannot resolve it (e.g. "What are the penalties?" without specifying for what offence). Do NOT ask for clarification when: (a) the excerpts provide a clear single answer, (b) the question uses "my" or "I" but refers to a general process (e.g. "How do I file a patent?"), or (c) the topic has one standard answer in the corpus. If in doubt, answer from the excerpts.
2. VERBATIM ONLY: Copy the exact sentence(s) from the excerpts word-for-word. Do NOT add any introductory phrase (e.g. do not write "According to the document..." or "The text states..."). Do NOT add any connecting words, commentary, transitions, or explanations of your own. Your answer body must contain ONLY text that appears character-for-character in the excerpts.
3. If the answer spans multiple excerpts, paste each relevant passage verbatim one after the other separated by a blank line. Zero added words between them.
4. Include ALL sentences from the excerpt that are needed to fully answer the question — do not truncate.
5. OUT-OF-SCOPE CHECK: If the excerpts do not directly address the question (they only mention the topic in passing or cover a different subject), respond only with — "I don't have information about this in the available documents."
6. If the information is NOT found in any excerpt: respond only with — "I don't have information about this in the available documents."
7. End every answer with: **Sources:** listing ONLY the actual source filename and page numbers from the excerpts you used (e.g. **Sources:** AA00007386_00001.pdf, pp. 28, 130). Never invent or guess page numbers — use only those shown in the [Excerpt | source | page X] labels above."""

SYSTEM_PROMPT = _load_system_prompt()

# ── Yes/No dynamic prompt ──────────────────────────────────────────────────────

_YES_NO_RE = re.compile(
    r'^(is|are|was|were|can|could|does|do|did|has|have|had|will|would|should|may|might|must|am)\b',
    re.IGNORECASE
)

def _is_yes_no(question: str) -> bool:
    return bool(_YES_NO_RE.match(question.strip()))


# ── Query reformulation (multi-turn memory) ────────────────────────────────────

REWRITE_PROMPT = """\
You are a query rewriter. Given a conversation history and the user's latest message, rewrite the latest message as a fully self-contained search query that can be understood without the history.
- Keep it concise (one sentence).
- Resolve all pronouns and references using the history.
- If the message is already self-contained, return it unchanged.
- Return ONLY the rewritten query, no explanation.
"""

def reformulate_query(question: str, history: list) -> str:
    """Rewrite `question` into a standalone query using conversation history."""
    if not history:
        return question
    turns = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in history[-MAX_HISTORY * 2:]
    )
    r = safe_chat(
        [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"Conversation so far:\n{turns}\n\nLatest message: {question}\n\nRewritten query:"},
        ],
        model=CHAT_MODEL,
        temperature=0,
        max_tokens=80,
    )
    if r is None:
        return question
    track_cost(r, script="main")
    return r.choices[0].message.content.strip() or question


# ── Index loading ──────────────────────────────────────────────────────────────

if not os.path.exists("data/my_index.faiss"):
    sys.exit("ERROR: Run  python app/build_index.py  first.")

index = faiss.read_index("data/my_index.faiss")
with open("data/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
print(f"Ready — {len(chunks)} chunks indexed (FAISS + BM25).")


def reload_index():
    """Hot-reload FAISS index, chunks, and BM25 after an incremental update.

    Called after add_web_results_to_index() so the running app immediately
    sees the new vectors without a full restart.
    """
    global index, chunks, bm25
    index = faiss.read_index("data/my_index.faiss")
    with open("data/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    print(f"[reload_index] {len(chunks)} chunks now in memory.")


# ── Hybrid retrieval (FAISS + BM25 → RRF) ────────────────────────────────────

def hybrid_retrieve(query: str, k: int = TOP_K) -> list[tuple[dict, float]]:
    """Reciprocal Rank Fusion over FAISS (semantic) + BM25 (keyword) results."""
    emb_resp = client.embeddings.create(model=EMBED_MODEL, input=query)
    track_cost(emb_resp, is_embedding=True, script="main")
    q_vec = np.array(emb_resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_vec)
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


# ── RAG pipeline ───────────────────────────────────────────────────────────────

REFUSAL_PHRASES = (
    "i don't have information", "not in the available documents",
    "cannot find", "no information", "outside the scope",
    "not covered", "i don't know", "unable to find",
)

def rag_answer(question, history, session_cost):
    if not question.strip():
        return history, f"**Session cost:** ${session_cost:.4f}  \n**Last query:** $0.0000", session_cost, "", ""

    retrieval_query = reformulate_query(question, history)
    hits = hybrid_retrieve(retrieval_query)
    best_score = hits[0][1] if hits else 0

    # ── Confidence threshold ──────────────────────────────────────────────────
    low_confidence = best_score < RRF_THRESHOLD

    context = "\n\n---\n\n".join(
        f"[Excerpt {i+1} | {c['source']} | page {c['page']}]\n{c['text']}"
        for i, (c, _) in enumerate(hits)
    )

    # ── Dynamic yes/no prompt ─────────────────────────────────────────────────
    user_content = f"Context:\n\n{context}\n\nQuestion: {question}"
    if _is_yes_no(question):
        user_content += "\n\nIMPORTANT: Start your answer with Yes or No (as the very first word), then cite the exact supporting sentence verbatim."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in (history or [])[-MAX_HISTORY * 2:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    r = safe_chat(messages, model=CHAT_MODEL, temperature=0, max_tokens=700)
    if r is None:
        return history, f"**Session cost:** ${session_cost:.4f}  \n**Last query:** $0.0000", session_cost, "", ""
    track_cost(r, script="main")
    answer = r.choices[0].message.content or ""

    q_cost = (r.usage.prompt_tokens * COSTS["input"] + r.usage.completion_tokens * COSTS["output"]) / 1e6
    q_cost += len(enc.encode(question)) * COSTS["embed"] / 1e6
    new_cost = session_cost + q_cost

    # ── Sources panel ─────────────────────────────────────────────────────────
    seen, src_lines = set(), []
    for c, _ in hits:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            src_lines.append(f"**{c['source']}**, p. {c['page']}")

    query_note     = f"\n\n*Search query:* `{retrieval_query}`" if retrieval_query != question else ""
    confidence_note = "\n\n⚠️ *Low confidence — both retrievers returned weak matches. Answer may be unreliable.*" if low_confidence else ""
    sources_md = (
        "**Retrieved excerpts:**\n\n"
        + "\n".join(f"{i+1}. {s}" for i, s in enumerate(src_lines))
        + f"\n\n*Best RRF score: {best_score:.4f}*{query_note}{confidence_note}"
    )
    cost_md = f"**Session cost:** ${new_cost:.4f}  \n**Last query:** ${q_cost:.4f}"

    # ── LLM-as-Judge ─────────────────────────────────────────────────────────
    is_refusal = any(p in answer.lower() for p in REFUSAL_PHRASES)
    if is_refusal:
        judge_md = "### N/A — Refusal answer, no faithfulness check needed."
    else:
        verdict, reasoning = judge_faithfulness(question, answer, context)
        icon = "✅" if verdict == "faithful" else "❌"
        label = "Faithful" if verdict == "faithful" else "Unfaithful"
        judge_md = f"### {icon} {label}\n\n<details><summary>View reasoning</summary>\n\n{reasoning}\n\n</details>"

    updated_history = (history or []) + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]
    return updated_history, cost_md, new_cost, sources_md, judge_md


# ── Feedback logging ──────────────────────────────────────────────────────────

def log_feedback(data: gr.LikeData):
    os.makedirs("data", exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "index":     data.index,
        "liked":     data.liked,
        "message":   data.value if isinstance(data.value, str) else str(data.value),
    }
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ── Export conversation ───────────────────────────────────────────────────────

def export_conversation(history):
    if not history:
        return None
    lines = []
    for msg in history:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"**{role}:** {msg['content']}\n")
    text = "\n---\n\n".join(lines)
    path = f"data/conversation_{int(time.time())}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ── Gradio UI ─────────────────────────────────────────────────────────────────

EXAMPLES = [
    "What is a contract?",
    "What is the difference between civil law and criminal law?",
    "How does common law differ from civil law?",
    "What is a tort?",
    "Is remote attendance allowed at court hearings?",
    "What is the weather in Paris today?",
    "Tell me about the requirements.",
]

with gr.Blocks(title="Business Law RAG Assistant") as demo:
    gr.Markdown(
        "# 📚 Business Law RAG Assistant\n"
        "Answers grounded in the corpus with source citations.\n"
        "> *GPT-4o-mini · text-embedding-3-small · FAISS + BM25 hybrid*"
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation", height=480,
                buttons=["copy"],
            )
            with gr.Row():
                msg_box  = gr.Textbox(placeholder="Ask a question about business law…", label="", container=False, scale=5)
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)
            with gr.Row():
                clear_btn  = gr.Button("🗑  Clear", variant="secondary", size="sm")
                export_btn = gr.Button("💾  Export chat", variant="secondary", size="sm")
            export_file = gr.File(label="Download", visible=False)
            gr.Examples(examples=EXAMPLES, inputs=msg_box, label="Example questions")

        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### 💰 Cost tracker")
            cost_display    = gr.Markdown("**Session cost:** $0.0000  \n**Last query:** $0.0000")
            gr.Markdown("---\n### 📄 Retrieved sources")
            sources_display = gr.Markdown("*Sources will appear here after your first query.*")
            gr.Markdown("---\n### 🔍 LLM-as-Judge")
            judge_display   = gr.Markdown("*Faithfulness verdict will appear here after your first query.*")
            gr.Markdown(
                "---\n**Model:** gpt-4o-mini  \n**Embeddings:** text-embedding-3-small  \n"
                "**Retrieval:** FAISS + BM25 → RRF  \n**Top-K:** 5 chunks  \n**Corpus:** Business Law (469 pages)"
            )

    cost_state = gr.State(0.0)
    outputs    = [chatbot, cost_display, cost_state, sources_display, judge_display]

    msg_box.submit(rag_answer, [msg_box, chatbot, cost_state], outputs).then(lambda: "", outputs=msg_box)
    send_btn.click(rag_answer, [msg_box, chatbot, cost_state], outputs).then(lambda: "", outputs=msg_box)
    clear_btn.click(
        lambda: ([], "**Session cost:** $0.0000  \n**Last query:** $0.0000", 0.0,
                 "*Sources will appear here after your first query.*",
                 "*Faithfulness verdict will appear here after your first query.*"),
        outputs=outputs,
    )
    export_btn.click(export_conversation, inputs=chatbot, outputs=export_file).then(
        lambda: gr.File(visible=True), outputs=export_file
    )
    chatbot.like(log_feedback, None, None)

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=False, theme=gr.themes.Soft(primary_hue="blue"))
