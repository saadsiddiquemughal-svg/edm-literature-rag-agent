"""
Pulls papers from arXiv matching EDM storage-ring search terms, saves the
PDFs and metadata locally. 
"""

import time
import json
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API = "http://export.arxiv.org/api/query"
PAPERS_DIR = Path("data/papers")
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# arXiv rate-limits harder if you look like an anonymous script - a
# descriptive User-Agent and a real gap between requests avoids most

HEADERS = {"User-Agent": "edm-rag-personal-project/1.0 (research literature tool)"}


SEARCH_QUERIES = [
    "electric dipole moment of charged particles in storage rings",
    "axion particles in storage rings",
    "frozen spin technique for protons and deuterons in storage rings",
    "COSY spin coherence time of proton and deuteron beams",
    "proton electric dipole moment measurement",
    "systematic errors in EDM storage ring",
    "beam dynamics simulation in EDM storage ring",
]

MAX_RESULTS_PER_QUERY = 5
SECONDS_BETWEEN_REQUESTS = 5  
MAX_RETRIES = 4

NS = {"atom": "http://www.w3.org/2005/Atom"}


def request_with_retry(url):
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < MAX_RETRIES - 1:
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s...
                print(f"    got {e.code}, waiting {wait}s before retry "
                      f"({attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("out of retries")


def search_arxiv(query, max_results=15):
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    xml_data = request_with_retry(f"{ARXIV_API}?{params}")
    root = ET.fromstring(xml_data)
    entries = []
    for entry in root.findall("atom:entry", NS):
        arxiv_id = entry.find("atom:id", NS).text.split("/abs/")[-1]
        title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
        pdf_url = None
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
        published = entry.find("atom:published", NS).text[:10]
        entries.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "summary": summary,
            "pdf_url": pdf_url,
            "published": published,
        })
    return entries


def download_pdf(entry):
    out_path = PAPERS_DIR / f"{entry['arxiv_id'].replace('/', '_')}.pdf"
    if out_path.exists():
        return out_path
    data = request_with_retry(entry["pdf_url"])
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def main():
    seen = {}
    for q in SEARCH_QUERIES:
        print(f"searching: {q}")
        try:
            results = search_arxiv(q, MAX_RESULTS_PER_QUERY)
            for r in results:
                seen[r["arxiv_id"]] = r
            print(f"  {len(results)} results")
        except Exception as e:
            print(f"  failed: {e}")
        time.sleep(SECONDS_BETWEEN_REQUESTS)

    print(f"\n{len(seen)} unique papers found, downloading PDFs...")
    metadata = []
    for arxiv_id, entry in seen.items():
        try:
            path = download_pdf(entry)
            entry["local_path"] = str(path)
            metadata.append(entry)
            print(f"  ok: {entry['title'][:70]}")
        except Exception as e:
            print(f"  failed to download {arxiv_id}: {e}")
        time.sleep(SECONDS_BETWEEN_REQUESTS)

    with open(PAPERS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\ndone. {len(metadata)} papers saved to {PAPERS_DIR}/")


if __name__ == "__main__":
    main()
