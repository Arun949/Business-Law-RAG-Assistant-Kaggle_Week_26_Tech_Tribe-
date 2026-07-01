"""
gradio_app.py — Gradio chatbot UI for the 3-agent pipeline.

Run from project root:
    python app/gradio_app.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
from agents import run_pipeline, hitl_more_research, hitl_more_web
from llm_judge import judge_faithfulness
from mcp_server import create_markdown_report, add_web_results_to_index
from main import reload_index
from analytics import build_analytics_tab

EXAMPLES = [
    "What are the first ten amendments to the US Constitution called?",
    "What is precedent in common law, and what is the benefit of this system?",
    "Who is the current Chief Justice of the United States?",
    "What is a contract?",
    "What is a tort?",
    "generate a report on contract law",
]

REFUSAL_PHRASES = (
    "i don't have information", "not in the available documents",
    "cannot find", "no information", "outside the scope",
    "not covered", "i don't know", "unable to find",
)

_PLACEHOLDER_CORPUS = "*Corpus excerpts will appear here after your first query.*"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_corpus_panel(corpus_chunks: list[dict]) -> str:
    if not corpus_chunks:
        return _PLACEHOLDER_CORPUS
    lines = ["### 📚 Retrieved Corpus Excerpts\n"]
    for i, c in enumerate(corpus_chunks, 1):
        lines.append(
            f"**Excerpt {i} — p.{c['page']}** *(RRF: {c['score']})*\n"
            f"<details><summary>View info</summary>\n\n{c['text']}\n\n</details>"
        )
    return "\n\n".join(lines)


def _strip_sources_line(answer: str) -> str:
    """Remove the Sources line — the judge only needs the answer paragraph."""
    import re
    stripped = re.split(r"(?i)\n*Sources\s*:", answer, maxsplit=1)[0]
    return stripped.strip()



def _run_judge(query: str, answer: str, corpus_context: str) -> str:
    is_refusal = any(p in answer.lower() for p in REFUSAL_PHRASES)
    if is_refusal:
        return "### N/A — Refusal answer, no faithfulness check needed."
    no_corpus = (not corpus_context or corpus_context.strip() == "No corpus excerpts found.")
    if no_corpus:
        return "### N/A — No corpus content to evaluate."
    clean_answer = _strip_sources_line(answer)
    verdict, reasoning = judge_faithfulness(query, clean_answer, corpus_context)
    icon  = "✅" if verdict == "faithful" else "❌"
    label = "Faithful" if verdict == "faithful" else "Unfaithful"
    return (
        f"### {icon} {label}\n\n"
        f"<details><summary>View reasoning</summary>\n\n{reasoning}\n\n</details>"
    )


# ── Chat handler ──────────────────────────────────────────────────────────────

def chat(query: str, history: list, session_cost: float, hitl_state: dict, use_web: bool, session_log: list):
    if not query.strip():
        return (history,
                f"**Session cost:** ${session_cost:.4f}  \n**Last query:** $0.0000",
                session_cost,
                _PLACEHOLDER_CORPUS,
                "*Ask a question to see the faithfulness verdict.*",
                hitl_state,
                "",
                session_log)

    final_answer, corpus_chunks, web_results, query_cost, corpus_context = run_pipeline(query, use_web=use_web)
    display_answer = final_answer

    new_cost  = session_cost + query_cost
    cost_md   = f"**Session cost:** ${new_cost:.4f}  \n**Last query:** ${query_cost:.4f}"
    corpus_md = _format_corpus_panel(corpus_chunks)
    judge_md  = _run_judge(query, final_answer, corpus_context)

    new_hitl_state = {
        "query":         query,
        "corpus_chunks": corpus_chunks,
        "web_results":   web_results,
        "last_answer":   display_answer,
    }

    faithful = "faithful" if "✅" in judge_md else "unfaithful" if "❌" in judge_md else "n/a"
    new_session_log = session_log + [{
        "question":  query,
        "best_rrf":  corpus_chunks[0]["score"] if corpus_chunks else 0.0,
        "faithful":  faithful,
        "cost":      query_cost,
        "ts":        time.strftime("%Y-%m-%d %H:%M:%S"),
    }]

    history = history + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": display_answer},
    ]

    return history, cost_md, new_cost, corpus_md, judge_md, new_hitl_state, "", new_session_log


# ── HITL: Approve → generate report + preview ────────────────────────────────

def hitl_approve(hitl_state: dict):
    """Approve the latest answer: save as .md, show preview, reveal Add-to-DB button."""
    if not hitl_state.get("query"):
        return gr.File(visible=False), gr.update(visible=False, value="*No answer to approve yet.*"), gr.update(visible=False)

    query       = hitl_state["query"]
    last_answer = hitl_state.get("last_answer", "")

    os.makedirs("data", exist_ok=True)
    filename = f"data/report_{int(time.time())}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(
            f"# Approved Report\n\n"
            f"**Query:** {query}\n\n"
            f"## Answer\n\n{last_answer}\n"
        )

    preview = f"### 📄 Report Preview\n\n**Query:** {query}\n\n---\n\n{last_answer}"
    return filename, gr.update(visible=True, value=preview), gr.update(visible=True)


# ── HITL: Add web results to database ────────────────────────────────────────

def hitl_add_to_db(hitl_state: dict):
    """Embed web results and append them incrementally to the FAISS index."""
    web_results = hitl_state.get("web_results", [])
    if not web_results:
        return "⚠️ No web results in the last query to add.", gr.update(visible=False)

    status = add_web_results_to_index(web_results)
    reload_index()   # hot-reload globals in main.py so next query uses updated index
    return f"✅ {status}", gr.update(visible=False)


# ── HITL: More Corpus ─────────────────────────────────────────────────────────

def hitl_corpus_fn(history: list, session_cost: float, hitl_state: dict):
    """Re-run Agent 1 results through Agent 3 with all 5 chunks."""
    if not hitl_state.get("query"):
        return history, f"**Session cost:** ${session_cost:.4f}", session_cost, "*No query yet.*", gr.update(visible=False)

    query         = hitl_state["query"]
    corpus_chunks = hitl_state["corpus_chunks"]
    web_results   = hitl_state["web_results"]

    answer, cost, corpus_context = hitl_more_research(query, corpus_chunks, web_results)
    new_cost = session_cost + cost
    judge_md = _run_judge(query, answer, corpus_context)

    history = history + [
        {"role": "user",      "content": "📚 *Requested more corpus research (all chunks)*"},
        {"role": "assistant", "content": answer},
    ]
    return history, f"**Session cost:** ${new_cost:.4f}  \n**Last query:** ${cost:.4f}", new_cost, judge_md, gr.update(visible=False)


# ── HITL: More Web ────────────────────────────────────────────────────────────

def hitl_web_fn(history: list, session_cost: float, hitl_state: dict):
    """Re-run Agent 2 (fresh web search) + Agent 3."""
    if not hitl_state.get("query"):
        return history, f"**Session cost:** ${session_cost:.4f}", session_cost, "*No query yet.*", gr.update(visible=False)

    query         = hitl_state["query"]
    corpus_chunks = hitl_state["corpus_chunks"]

    answer, _, cost, corpus_context = hitl_more_web(query, corpus_chunks)
    new_cost = session_cost + cost
    judge_md = _run_judge(query, answer, corpus_context)

    history = history + [
        {"role": "user",      "content": "🌐 *Requested more web research (fresh search)*"},
        {"role": "assistant", "content": answer},
    ]
    return history, f"**Session cost:** ${new_cost:.4f}  \n**Last query:** ${cost:.4f}", new_cost, judge_md, gr.update(visible=False)


# ── HITL: Rephrase (re-run full pipeline) ────────────────────────────────────

def hitl_rephrase_fn(history: list, session_cost: float, hitl_state: dict):
    """Re-run the full 3-agent pipeline to get a fresh answer."""
    if not hitl_state.get("query"):
        return history, f"**Session cost:** ${session_cost:.4f}", session_cost, _PLACEHOLDER_CORPUS, "*No query yet.*", hitl_state

    query = hitl_state["query"]
    final_answer, corpus_chunks, web_results, query_cost, corpus_context = run_pipeline(query)
    display_answer = final_answer

    new_cost  = session_cost + query_cost
    corpus_md = _format_corpus_panel(corpus_chunks)
    judge_md  = _run_judge(query, final_answer, corpus_context)

    new_hitl_state = {
        "query":         query,
        "corpus_chunks": corpus_chunks,
        "web_results":   web_results,
        "last_answer":   display_answer,
    }

    history = history + [
        {"role": "user",      "content": "🔄 *Requested a full rephrase (all agents re-run)*"},
        {"role": "assistant", "content": display_answer},
    ]
    return history, f"**Session cost:** ${new_cost:.4f}  \n**Last query:** ${query_cost:.4f}", new_cost, corpus_md, judge_md, new_hitl_state


# ── Export ────────────────────────────────────────────────────────────────────

def export_chat(history: list):
    if not history:
        return None
    lines = []
    for msg in history:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"**{role}:** {msg['content']}\n")
    text = "\n---\n\n".join(lines)
    path = f"data/conversation_{int(time.time())}.txt"
    os.makedirs("data", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Multi-Agent RAG Assistant") as demo:

    cost_state        = gr.State(0.0)
    hitl_state        = gr.State({})
    use_web_state     = gr.State(False)   # AI Agent OFF by default
    session_log_state = gr.State([])

    with gr.Tabs():
        with gr.Tab("💬 Chat"):

            gr.Markdown(
                "# 🤖 Multi-Agent RAG Assistant\n"
                "**Agent 1** retrieves verbatim corpus excerpts · "
                "**Agent 2** searches the live web · "
                "**Agent 3** synthesizes the final answer\n"
                "> *GPT-4o-mini · FAISS + BM25 · DuckDuckGo + Wikipedia MCP*"
            )

            with gr.Row(equal_height=False):

                # ── Left: chat panel ─────────────────────────────────────────
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Conversation", height=400)

                    with gr.Row():
                        msg_box  = gr.Textbox(
                            placeholder="Ask a question…",
                            label="", container=False, scale=5,
                        )
                        send_btn = gr.Button("Send ➤", variant="primary", scale=1)

                    with gr.Row():
                        agent_toggle = gr.Button(
                            "🤖 AI Agent: OFF  (corpus only)",
                            variant="secondary",
                            size="sm",
                        )

                    # ── Human-in-the-Loop buttons ─────────────────────────────
                    gr.Markdown("**Human-in-the-Loop**")
                    with gr.Row():
                        approve_btn  = gr.Button("✅ Approve",  variant="primary",   size="sm")
                        rewrite_btn  = gr.Button("✏️ Rewrite",  variant="secondary", size="sm")
                        rephrase_btn = gr.Button("🔄 Rephrase", variant="secondary", size="sm")

                    # Rewrite sub-row (hidden until Rewrite is clicked)
                    with gr.Row(visible=False) as rewrite_row:
                        gr.Markdown("*Rewrite using more:*")
                        corpus_btn = gr.Button("📚 Corpus", variant="secondary", size="sm")
                        web_btn    = gr.Button("🌐 Web",    variant="secondary", size="sm")

                    with gr.Row():
                        clear_btn  = gr.Button("🗑  Clear",       variant="stop",      size="sm")
                        export_btn = gr.Button("💾  Export chat", variant="secondary", size="sm")

                    report_file    = gr.File(label="📄 Approved Report", visible=False)
                    report_preview = gr.Markdown(visible=False)
                    add_db_btn     = gr.Button("🗃️ Add Web Results to Database", variant="primary", size="sm", visible=False)
                    db_status      = gr.Markdown(visible=False)
                    export_file    = gr.File(label="💾 Download", visible=False)
                    gr.Examples(examples=EXAMPLES, inputs=msg_box, label="Example questions")

                # ── Right: info panel ────────────────────────────────────────
                with gr.Column(scale=1, min_width=280):
                    gr.Markdown("### 💰 Cost Tracker")
                    cost_display = gr.Markdown("**Session cost:** $0.0000  \n**Last query:** $0.0000")

                    gr.Markdown("---")
                    corpus_display = gr.Markdown(_PLACEHOLDER_CORPUS)

                    gr.Markdown("---\n### 🔍 LLM-as-Judge")
                    judge_display = gr.Markdown("*Faithfulness verdict will appear here.*")

                    gr.Markdown(
                        "---\n"
                        "**Model:** gpt-4o-mini  \n"
                        "**Embeddings:** text-embedding-3-small  \n"
                        "**Retrieval:** FAISS + BM25 → RRF  \n"
                        "**Top-K:** 5 chunks"
                    )

        with gr.Tab("📊 Analytics"):
            build_analytics_tab(session_log_state)

    # ── Toggle handler ────────────────────────────────────────────────────────

    def _toggle_agent(current: bool):
        new_val = not current
        if new_val:
            label = "🤖 AI Agent: ON  (corpus + live web)"
            variant = "primary"
        else:
            label = "🤖 AI Agent: OFF  (corpus only)"
            variant = "secondary"
        return new_val, gr.update(value=label, variant=variant)

    agent_toggle.click(_toggle_agent, inputs=use_web_state, outputs=[use_web_state, agent_toggle])

    # ── Event wiring ──────────────────────────────────────────────────────────

    # Send / submit
    chat_inputs  = [msg_box, chatbot, cost_state, hitl_state, use_web_state, session_log_state]
    chat_outputs = [chatbot, cost_display, cost_state, corpus_display, judge_display, hitl_state, msg_box, session_log_state]
    msg_box.submit(chat, chat_inputs, chat_outputs)
    send_btn.click(chat,  chat_inputs, chat_outputs)

    # Approve → generate report + show preview + reveal Add-to-DB button
    approve_btn.click(
        hitl_approve,
        [hitl_state],
        [report_file, report_preview, add_db_btn],
    )

    # Add to Database → embed web results, update index, show status
    add_db_btn.click(
        hitl_add_to_db,
        [hitl_state],
        [db_status, add_db_btn],
    ).then(lambda: gr.update(visible=True), outputs=db_status)

    # Rewrite → show sub-row
    rewrite_btn.click(lambda: gr.update(visible=True), outputs=rewrite_row)

    # Rewrite → Corpus
    hitl_row_outputs = [chatbot, cost_display, cost_state, judge_display, rewrite_row]
    corpus_btn.click(hitl_corpus_fn, [chatbot, cost_state, hitl_state], hitl_row_outputs)

    # Rewrite → Web
    web_btn.click(hitl_web_fn, [chatbot, cost_state, hitl_state], hitl_row_outputs)

    # Rephrase → full re-run
    rephrase_outputs = [chatbot, cost_display, cost_state, corpus_display, judge_display, hitl_state]
    rephrase_btn.click(hitl_rephrase_fn, [chatbot, cost_state, hitl_state], rephrase_outputs)

    # Clear
    clear_btn.click(
        lambda: ([], "**Session cost:** $0.0000  \n**Last query:** $0.0000", 0.0,
                 _PLACEHOLDER_CORPUS, "*Faithfulness verdict will appear here.*", {},
                 gr.update(visible=False)),
        outputs=[chatbot, cost_display, cost_state, corpus_display, judge_display, hitl_state, rewrite_row],
    )

    # Export chat
    export_btn.click(export_chat, inputs=chatbot, outputs=export_file).then(
        lambda: gr.File(visible=True), outputs=export_file
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        share=False,
        theme=gr.themes.Soft(primary_hue="blue"),
        allowed_paths=["data/sources"],
    )
