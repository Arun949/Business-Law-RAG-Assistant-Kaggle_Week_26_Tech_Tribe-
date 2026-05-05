import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from dotenv import load_dotenv
from main import hybrid_retrieve
from mcp_server import search_web_with_sources, create_markdown_report
from cost_tracker import track_cost

load_dotenv()
_client = OpenAI()
CHAT_MODEL = "gpt-4o-mini"

SYNTHESIZER_PROMPT = """\
You are a Synthesizer agent. You receive:
1. The user's original query
2. One verbatim excerpt from a private document corpus, labelled [Excerpt 1 | source | page X]
3. Live web search findings with source titles and URLs

Output format — two parts only, nothing else:

[PART 1] Write the corpus sentences first. Pick the 2–3 sentences from the corpus excerpt that most directly answer the query. Copy them exactly, word-for-word. Then, without any line break or heading, continue in the same paragraph with 1–2 verbatim sentences from the web findings. The result must be one unbroken paragraph: corpus content first, web content appended after.

[PART 2] On a new line, write "Sources: " followed by the page number used (e.g. p.172) and any web URLs, separated by " · ".

STRICT RULES:
- Do NOT use any headers, labels, or prefixes (no "PART 1", "CORPUS:", "WEB:", "Web search adds:", etc.).
- Corpus sentences MUST come before web sentences — never swap the order.
- Zero invented words. Every sentence must be copied verbatim from the source material.
- If corpus does not address the query: write the paragraph using only web findings.
- If neither source has info: respond only with "I don't have information about this in the available documents."
"""


OUT_OF_SCOPE_MSG = (
    "I don't have information about this.\n"
    "I can only answer questions related to business and contract law, "
    "legal concepts, rights, obligations, torts, or legal procedures."
)

DOMAIN_ROUTER_PROMPT = """\
You are a query router for a business and contract law document corpus.
Answer with a single word — 'yes' or 'no':
Is the following question CLEARLY AND OBVIOUSLY unrelated to law, legal concepts, contracts, torts, business, rights, obligations, or any academic/professional subject?
Only answer 'yes' if the question is about something like weather, sports scores, cooking, entertainment, or personal chit-chat.
If the question is vague, general, or could plausibly relate to law or business — answer 'no'."""


# ── Router: domain relevance check ───────────────────────────────────────────

def _is_in_domain(query: str) -> bool:
    """Return True if the query is relevant to the corpus domain.

    Uses an inverted check: only blocks queries that are clearly off-topic
    (weather, sports, cooking, etc.). Vague or general questions pass through.
    """
    r = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": DOMAIN_ROUTER_PROMPT},
            {"role": "user",   "content": query},
        ],
        temperature=0,
        max_tokens=5,
    )
    is_clearly_off_topic = r.choices[0].message.content.strip().lower().startswith("yes")
    return not is_clearly_off_topic


# ── Agent 1: Internal Researcher ─────────────────────────────────────────────

def _agent1_internal_researcher(query: str) -> list[dict]:
    """Retrieve top-matching corpus chunks verbatim.

    Returns list of {text, source, page, score}.
    """
    hits = hybrid_retrieve(query)
    return [
        {"text": chunk["text"], "source": chunk["source"], "page": chunk["page"], "score": round(score, 4)}
        for chunk, score in hits
    ] if hits else []


# ── Agent 2: External Fact-Checker ───────────────────────────────────────────

def _agent2_external_fact_checker(query: str) -> list[dict]:
    """Search the live web via MCP.

    Returns list of {url, title, snippet}.
    """
    return search_web_with_sources(query)


# ── Agent 3: Synthesizer ──────────────────────────────────────────────────────

def _agent3_synthesizer(
    query: str,
    corpus_chunks: list[dict],
    web_results: list[dict],
    extra_instruction: str = "",
) -> tuple[str, float, str]:
    """Combine corpus excerpts and web results into a single coherent answer.

    Returns (answer, query_cost_usd, corpus_context_str).
    extra_instruction is appended to the user message for HITL rewrite/research modes.
    """
    corpus_text = (
        "\n\n---\n\n".join(
            f"[Excerpt {i+1} | {c['source']} | page {c['page']}]\n{c['text']}"
            for i, c in enumerate(corpus_chunks)
        )
        if corpus_chunks else "No corpus excerpts found."
    )
    web_text = (
        "\n\n".join(
            f"[{r['title']} | {r['url']}]\n{r['snippet']}"
            for r in web_results
        )
        if web_results else "No web results found."
    )

    user_content = (
        f"User query: {query}\n\n"
        f"--- Corpus excerpts (verbatim) ---\n{corpus_text}\n\n"
        f"--- Live web findings ---\n{web_text}"
    )
    if extra_instruction:
        user_content += f"\n\nHUMAN FEEDBACK: {extra_instruction}"

    r = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIZER_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=600,
    )
    cost = track_cost(r, script="agents")
    return r.choices[0].message.content.strip(), cost, corpus_text


# ── HITL helpers (called from UI on human feedback) ───────────────────────────

def hitl_more_research(
    query: str, corpus_chunks: list[dict], web_results: list[dict]
) -> tuple[str, float, str]:
    """Human-in-the-Loop: re-synthesize using ALL retrieved corpus chunks
    (instead of just top-1) to give a more thorough answer."""
    return _agent3_synthesizer(
        query,
        corpus_chunks,          # pass all 5 chunks
        web_results,
        extra_instruction="The human requested more thorough research. Use ALL provided corpus excerpts.",
    )


def hitl_rewrite(
    query: str, corpus_chunks: list[dict], web_results: list[dict]
) -> tuple[str, float, str]:
    """Human-in-the-Loop: ask Agent 3 to rewrite the answer more clearly."""
    return _agent3_synthesizer(
        query,
        corpus_chunks,
        web_results,
        extra_instruction="The human requested a rewrite. Make the answer clearer, more concise, and better structured.",
    )


def hitl_more_web(
    query: str, corpus_chunks: list[dict]
) -> tuple[str, list[dict], float, str]:
    """Human-in-the-Loop: re-run Agent 2 (fresh web search) then Agent 3.

    Returns (answer, new_web_results, cost, corpus_context).
    """
    new_web_results = _agent2_external_fact_checker(query)
    answer, cost, corpus_context = _agent3_synthesizer(
        query,
        corpus_chunks[:1],
        new_web_results,
        extra_instruction="The human requested more web research. Prioritise the fresh web findings.",
    )
    return answer, new_web_results, cost, corpus_context


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_pipeline(query: str, use_web: bool = True) -> tuple[str, list[dict], list[dict], float, str]:
    """Run the agent pipeline.

    Args:
        query:   the user's question
        use_web: when True, Agent 2 searches the live web and Agent 3 synthesizes
                 corpus + web results. When False, only corpus chunks are used
                 (no web call, lower cost).

    Returns:
        final_answer    (str)        — synthesized answer for the chatbot
        corpus_chunks   (list[dict]) — [{text, source, page, score}]
        web_results     (list[dict]) — [{url, title, snippet}]
        query_cost      (float)      — USD cost of this query (Agent 3 only)
        corpus_context  (str)        — raw context string fed to Agent 3 (for LLM judge)
    """
    corpus_chunks = _agent1_internal_researcher(query)

    # Route: skip web entirely if the query is outside the corpus domain
    if not _is_in_domain(query):
        return OUT_OF_SCOPE_MSG, corpus_chunks, [], 0.0, ""

    if not use_web:
        # Corpus-only mode: synthesize with no web results
        final_answer, query_cost, corpus_context = _agent3_synthesizer(
            query, corpus_chunks[:1], []
        )
        return final_answer, corpus_chunks, [], query_cost, corpus_context

    # Full pipeline: Agent 2 searches the web, Agent 3 synthesizes both
    web_results = _agent2_external_fact_checker(query)
    final_answer, query_cost, corpus_context = _agent3_synthesizer(query, corpus_chunks[:1], web_results)

    if "report" in query.lower():
        path = create_markdown_report(
            content=f"# Report\n\n**Query:** {query}\n\n{final_answer}",
            filename="report.md",
        )
        print(f"[MCP] {path}")

    return final_answer, corpus_chunks, web_results, query_cost, corpus_context


def agent(query: str) -> str:
    """Terminal entry-point — returns only the final answer."""
    print("[Agent 1] Internal Researcher — searching corpus…")
    print("[Agent 2] External Fact-Checker — searching live web via MCP…")
    print("[Agent 3] Synthesizer — combining outputs…")
    final_answer, _, _, _, _ = run_pipeline(query)
    return final_answer
