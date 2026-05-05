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
# ── Chunking (sentence-boundary aware) ────────────────────────────────────────
# Splits on sentence boundaries first so chunks never cut mid-sentence,
# then groups sentences until CHUNK_TARGET_CHARS is reached.
# Overlap carries the last few sentences into the next chunk for context.

_SENT_RE = re.compile(r'(?<=[.!?])\s+')

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on .!? boundaries."""
    sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    # If no sentence boundaries found, fall back to the whole paragraph as one unit
    return sentences if sentences else [text.strip()]

def chunk_entry(entry: dict) -> list[dict]:
    text = entry["text"]
    if len(text) < MIN_CHUNK_CHARS:
        return []

    # Split into paragraphs first, then sentences within each paragraph
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        sentences.extend(_split_sentences(para))

    if not sentences:
        return []

    chunks: list[dict] = []
    window: list[str] = []
    window_len = 0

    for sent in sentences:
        if window_len + len(sent) > CHUNK_TARGET_CHARS and window:
            chunk_text = " ".join(window)
            if len(chunk_text.strip()) >= MIN_CHUNK_CHARS:
                chunks.append({
                    "text": chunk_text,
                    "source": entry["source"],
                    "page": entry["page"],
                })
            # Overlap: carry last sentences whose total length ≤ OVERLAP_CHARS
            overlap: list[str] = []
            overlap_len = 0
            for s in reversed(window):
                if overlap_len + len(s) <= OVERLAP_CHARS:
                    overlap.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            window = overlap
            window_len = overlap_len

        window.append(sent)
        window_len += len(sent)

    # Flush remaining
    if window:
        chunk_text = " ".join(window)
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
# ── Main 
def main():
    corpus_path = "data/corpus.json"
    index_path  = "data/my_index.faiss"
    chunks_path = "data/chunks.json"
    # ── 1. Load corpus 
    print(f"Loading corpus from {corpus_path} ...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"  {len(corpus)} pages loaded")
    # ── 2. Chunk 
    print("\nChunking pages ...")
    all_chunks: list[dict] = []
    skipped = 0
    for entry in corpus:
        if entry.get("char_count", len(entry["text"])) < MIN_CHUNK_CHARS:
            skipped += 1
            continue
        all_chunks.extend(chunk_entry(entry))
    print(f"  Skipped {skipped} low-text pages (< {MIN_CHUNK_CHARS} chars)")
    print(f"  Total chunks produced: {len(all_chunks)}")
    if not all_chunks:
        print("ERROR: No chunks produced. Check corpus.json.")
        return
    # ── 3. Estimate cost 
    texts = [c["text"] for c in all_chunks]
    total_tokens = count_tokens(texts)
    estimated_cost = (total_tokens / 1_000_000) * COST_PER_1M_TOKENS
    print(f"\nTokens to embed : {total_tokens:,}")
    print(f"Estimated cost  : ${estimated_cost:.4f}")
    print()
    # ── 4. Embed 
    print("Embedding chunks ...")
    embeddings = embed_all(texts)
    # ── 5. Normalize → cosine similarity via IndexFlatIP 
    print("\nBuilding FAISS index (cosine similarity) ...")
    faiss.normalize_L2(embeddings)          # in-place L2 normalisation
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)          # inner product == cosine for unit vectors
    index.add(embeddings)
    print(f"  Index contains {index.ntotal} vectors (dim={dim})")
    # ── 6. Save 
    os.makedirs("app", exist_ok=True)
    faiss.write_index(index, index_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    actual_cost = (total_tokens / 1_000_000) * COST_PER_1M_TOKENS
    print(f"\n✅  Saved {index_path}")
    print(f"✅  Saved {chunks_path}  ({len(all_chunks)} chunks)")
    print(f"💰  Embedding cost: ${actual_cost:.4f}")
if __name__ == "__main__":
    main()
