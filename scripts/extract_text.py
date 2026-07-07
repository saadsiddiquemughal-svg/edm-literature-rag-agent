"""
Turns downloaded PDFs into overlapping text chunks ready for indexing.
Chunking by characters rather than tokens
"""

import json
import re
from pathlib import Path
import pdfplumber

PAPERS_DIR = Path("data/papers")
CHUNKS_PATH = Path("data/index/chunks.json")

CHUNK_SIZE = 1200      
CHUNK_OVERLAP = 200   


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            pages.append(t)
    return pages


def chunk_text(text, paper_meta, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 100: 
            chunks.append({
                "text": chunk,
                "arxiv_id": paper_meta["arxiv_id"],
                "title": paper_meta["title"],
            })
        start += chunk_size - overlap
    return chunks


def main():
    with open(PAPERS_DIR / "metadata.json") as f:
        papers = json.load(f)

    all_chunks = []
    for paper in papers:
        print(f"extracting: {paper['title'][:70]}")
        try:
            pages = extract_pdf_text(paper["local_path"])
            full_text = " ".join(pages)
            chunks = chunk_text(full_text, paper)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  failed: {e}")

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"\n{len(all_chunks)} chunks written to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
