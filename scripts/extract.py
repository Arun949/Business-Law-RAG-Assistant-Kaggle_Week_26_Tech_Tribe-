import fitz  # PyMuPDF
import json
import re
from pathlib import Path

PDF_DIR = Path("data/sources")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    return text.strip()


def extract_corpus(pdf_dir: Path):
    corpus = []
    low_text_log = []  # store flagged pages

    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text()
            text = clean_text(raw_text)
            char_count = len(text)

            # --- Detection (internal only, NOT saved in JSON) ---
            if char_count < 50:
                images = page.get_images(full=True)
                blocks = page.get_text("blocks")

                if images:
                    reason = "image/diagram"
                elif len(blocks) <= 1:
                    reason = "likely blank"
                else:
                    reason = "low text"

                low_text_log.append({
                    "source": pdf_path.name,
                    "page": page_num,
                    "reason": reason
                })

            # --- REQUIRED FORMAT ONLY ---
            corpus.append({
                "source": pdf_path.name,
                "page": page_num,
                "char_count": char_count,
                "text": text
            })

        doc.close()

    return corpus, low_text_log


def print_stats(corpus, low_text_log):
    sources = set(e["source"] for e in corpus)
    total_pages = len(corpus)
    total_chars = sum(e["char_count"] for e in corpus)
    avg_chars = total_chars / total_pages if total_pages else 0

    print("\n========== CORPUS STATS ==========")
    print(f"Documents      : {len(sources)}")
    print(f"Total pages    : {total_pages}")
    print(f"Total chars    : {total_chars:,}")
    print(f"Avg chars/page : {avg_chars:.0f}")

    print(f"\nLow-text pages (<50 chars): {len(low_text_log)}")

    if low_text_log:
        print("\nDetailed low-text pages:")
        for e in low_text_log:
            print(f"  - {e['source']} p.{e['page']} → {e['reason']}")

    print("===================================\n")


def main():
    corpus, low_text_log = extract_corpus(PDF_DIR)

    # Save corpus
    with open(OUTPUT_DIR / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Saved corpus.json ({len(corpus)} pages)")

    # Save sample
    with open(OUTPUT_DIR / "sample.json", "w", encoding="utf-8") as f:
        json.dump(corpus[:10], f, ensure_ascii=False, indent=2)
    print("Saved sample.json (first 10 pages)")

    print_stats(corpus, low_text_log)


if __name__ == "__main__":
    main()