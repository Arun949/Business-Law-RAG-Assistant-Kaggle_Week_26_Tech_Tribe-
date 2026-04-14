from dotenv import load_dotenv
load_dotenv()
import json
import os
import re
import time
import numpy as np
import faiss
import tiktoken
from openai import OpenAI
client = OpenAI()
enc = tiktoken.encoding_for_model("text-embedding-3-small")
# ── Configuration 
EMBED_MODEL        = "text-embedding-3-small"
CHUNK_TARGET_CHARS = 800   
OVERLAP_CHARS      = 120  
MIN_CHUNK_CHARS    = 50    
BATCH_SIZE         = 100   
COST_PER_1M_TOKENS = 0.020 
# ── Chunking 
def split_paragraphs(text: str) -> list[str]:
    """Split text on double newlines; return non-empty paragraphs."""
    return [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

def chunk_entry(entry: dict) -> list[dict]:
    text = entry["text"]
    if len(text) < MIN_CHUNK_CHARS:
        return []
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []
    chunks: list[dict] = []
    window: list[str] = []
    window_len = 0
    for para in paragraphs:
        if window_len + len(para) > CHUNK_TARGET_CHARS and window:
            chunk_text = "\n\n".join(window)
            if len(chunk_text.strip()) >= MIN_CHUNK_CHARS:
                chunks.append({
                    "text": chunk_text,
                    "source": entry["source"],
                    "page": entry["page"],
                })
            window = [window[-1]]
            window_len = len(window[0])
        window.append(para)
        window_len += len(para)

    # Flush remaining
    if window:
        chunk_text = "\n\n".join(window)
        if len(chunk_text.strip()) >= MIN_CHUNK_CHARS:
            chunks.append({
                "text": chunk_text,
                "source": entry["source"],
                "page": entry["page"],
            })

    return chunks
    
# ── Embedding 
def count_tokens(texts: list[str]) -> int:
    return sum(len(enc.encode(t)) for t in texts)
def embed_batch(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch of texts."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # Sort by index to guarantee order matches input
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
def embed_all(texts: list[str]) -> np.ndarray:
    """Embed all texts in batches; return (N, dim) float32 array."""
    all_embeddings: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        batch_embeddings = embed_batch(batch)
        all_embeddings.extend(batch_embeddings)
        end = min(start + BATCH_SIZE, total)
        print(f"  Embedded {end}/{total} chunks")
        # Brief pause to stay inside rate limits when corpus is large
        if end < total:
            time.sleep(0.2)
    return np.array(all_embeddings, dtype="float32")
