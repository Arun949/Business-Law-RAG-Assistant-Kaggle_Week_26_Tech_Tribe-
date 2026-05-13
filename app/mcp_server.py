import re
import json
import os
import numpy as np
import requests
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()
_openai = OpenAI()
EMBED_MODEL = "text-embedding-3-small"

# Create MCP server
mcp = FastMCP("RAG MCP Server")

_HEADERS = {"User-Agent": "RAGFactChecker/1.0 (educational hackathon project)"}
_HTML_TAG = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _HTML_TAG.sub("", text).strip()


# ---------------- TOOL 1: Live web search ----------------
@mcp.tool()
def search_web(query: str) -> str:
    """Search the live web for up-to-date information.
    Tries DuckDuckGo Instant Answer API first, then Wikipedia search API."""

    # --- DuckDuckGo Instant Answer API (no key required) ---
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            headers=_HEADERS,
            timeout=8,
        )
        data = resp.json()
        parts = []
        # Instant answer (e.g. "John Roberts" for a person query)
        if data.get("Answer"):
            parts.append(f"Answer: {data['Answer']}")
        if data.get("AbstractText"):
            parts.append(f"Summary: {data['AbstractText']}")
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(topic["Text"])
        if parts:
            return "\n\n".join(parts)
    except Exception:
        pass

    # --- Wikipedia Search API fallback (no key required) ---
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 3,
            },
            headers=_HEADERS,
            timeout=8,
        )
        results = resp.json().get("query", {}).get("search", [])
        if results:
            snippets = [
                f"{r['title']}: {_strip_html(r['snippet'])}"
                for r in results
            ]
            return "\n\n".join(snippets)
    except Exception:
        pass

    return f"No web results found for: {query}"


# ---------------- search_web_with_sources (used by agents, not MCP) ----------------

def search_web_with_sources(query: str) -> list[dict]:
    """Return a list of {url, title, snippet} dicts for use in the agent pipeline."""

    results: list[dict] = []

    # --- DDGS full-text search (primary) ---
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=3))
        for r in hits:
            results.append({
                "url":     r.get("href", ""),
                "title":   r.get("title", ""),
                "snippet": r.get("body", ""),
            })
        if results:
            return results
    except Exception:
        pass

    # --- Wikipedia Search API fallback ---
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list":   "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 3,
            },
            headers=_HEADERS,
            timeout=8,
        )
        for r in resp.json().get("query", {}).get("search", []):
            title = r["title"]
            results.append({
                "url":     f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "title":   title,
                "snippet": _strip_html(r["snippet"]),
            })
        if results:
            return results
    except Exception:
        pass

    return []


# ---------------- TOOL 2: Save markdown report ----------------
@mcp.tool()
def create_markdown_report(content: str, filename: str = "report.md") -> str:
    """Save a markdown-formatted report to disk."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Report saved to {filename}"


# ---------------- add_web_results_to_index (Python helper + MCP tool) --------

def add_web_results_to_index(web_results: list[dict]) -> str:
    """Incrementally embed web results and append them to the FAISS index.

    Only the NEW chunks are embedded — the existing corpus is untouched.
    Updates data/chunks.json and data/my_index.faiss in-place.

    Each web_result must have keys: url, title, snippet.
    Returns a status string.
    """
    chunks_path = "data/chunks.json"
    index_path  = "data/my_index.faiss"

    if not os.path.exists(chunks_path) or not os.path.exists(index_path):
        return "ERROR: Index not built yet. Run build_index.py first."

    # Build new chunks from web results
    new_chunks = []
    for r in web_results:
        text = f"{r.get('title', '')}: {r.get('snippet', '')}".strip()
        if len(text) >= 30:
            new_chunks.append({
                "text":   text,
                "source": r.get("url", "web"),
                "page":   0,
            })

    if not new_chunks:
        return "No valid web results to add."

    # Embed only the new chunks
    texts = [c["text"] for c in new_chunks]
    response = _openai.embeddings.create(model=EMBED_MODEL, input=texts)
    new_vecs  = np.array(
        [item.embedding for item in sorted(response.data, key=lambda x: x.index)],
        dtype="float32",
    )
    faiss.normalize_L2(new_vecs)

    # Append to existing FAISS index
    index = faiss.read_index(index_path)
    index.add(new_vecs)
    faiss.write_index(index, index_path)

    # Append to chunks.json
    with open(chunks_path, encoding="utf-8") as f:
        existing = json.load(f)
    existing.extend(new_chunks)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    return (
        f"✅ Added {len(new_chunks)} web chunk(s). "
        f"Index now has {index.ntotal} vectors / {len(existing)} chunks."
    )


@mcp.tool()
def add_to_database(snippets: list[str], urls: list[str], titles: list[str]) -> str:
    """Add web search results to the vector database incrementally.

    Embeds only the new content and appends it to the existing FAISS index
    and chunks.json without rebuilding from scratch.
    """
    web_results = [
        {"url": u, "title": t, "snippet": s}
        for s, u, t in zip(snippets, urls, titles)
    ]
    return add_web_results_to_index(web_results)


# Run server
if __name__ == "__main__":
    print("MCP Server running...")
    mcp.run()
