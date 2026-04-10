# EXTRACTION_REPORT.md

## Corpus Overview

| Metric | Value |
|---|---|
| Documents | 1 |
| Total pages | 469 |
| Total characters | 607,841 |
| Average characters/page | 1,296 |
| Low-text pages (< 50 chars) | 3 |

**Source file:** `AA00007386_00001.pdf`  
**Extraction library:** PyMuPDF (`pymupdf`)  

---

## Low-Text Pages

The following pages were flagged as containing fewer than 50 characters after cleaning:

| Page | Reason | Action taken |
|---|---|---|
| p.1 | Likely blank | Kept in corpus with empty text; excluded from meaningful content |
| p.4 | Likely blank | Kept in corpus with empty text; excluded from meaningful content |
| p.250 | Image/diagram detected | Kept in corpus; no text extractable via PyMuPDF — image content not captured |

These pages are included in `corpus.json` with their actual `char_count` so downstream filtering can handle them as needed.

---

## Issues Encountered

### 1. Blank pages (p.1, p.4)
The PDF opens with two blank or near-blank pages (likely cover/intentional blank pages). PyMuPDF returns no text blocks for these. They are retained in the corpus for page-index consistency but contribute no usable content.

### 2. Image/diagram page (p.250)
Page 250 contains an embedded image or diagram with no selectable text layer. PyMuPDF cannot extract text from rasterized images without OCR. The page is flagged and kept in the corpus with an empty text field. If the diagram content is important for retrieval, OCR post-processing (e.g., with `pytesseract` or `pymupdf` + `fitz` page rendering) would be required.

### 3. No garbled tables detected
The corpus does not appear to contain complex tables requiring `pdfplumber`. PyMuPDF output was spot-checked and found to be readable.

---

## Text Cleaning Applied

The following transformations were applied in `clean_text()`:

- Collapsed multiple spaces/tabs into a single space
- Reduced 3+ consecutive newlines to a maximum of 2
- Joined single line breaks within paragraphs into a single space (to fix mid-sentence line wrapping)
- Stripped leading/trailing whitespace

---

## Recommendations for RAG Pipeline

- **Filter low-text pages** (< 50 chars) at chunking time to avoid empty chunks polluting the vector index.
- **Page 250** may need manual review or OCR if its diagram is relevant to queries.
- Average chunk density (~1,296 chars/page) is reasonable for standard RAG chunking strategies (e.g., 512–1024 token chunks with overlap).
