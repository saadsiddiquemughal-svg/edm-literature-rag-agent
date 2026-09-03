"""
Search arXiv for EDM/storage-ring papers and download their PDFs.
The paper metadata is saved in data/papers/metadata.json.
"""

import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path


ARXIV_API = "http://export.arxiv.org/api/query"
PAPERS_DIR = Path("data/papers")

SEARCH_QUERIES = [
    "electric dipole moment storage ring",
    "frozen spin storage ring",
    "systematic errors EDM storage ring",
    "beam dynamics EDM storage ring",
]

MAX_RESULTS_PER_QUERY = 5
SECONDS_BETWEEN_REQUESTS = 5

HEADERS = {
    "User-Agent": "edm-rag-personal-project/1.0"
}

NS = {
    "atom": "http://www.w3.org/2005/Atom"
}


def fetch(url):
    """Download data from a URL."""
    request = urllib.request.Request(url, headers=HEADERS)

    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def search_arxiv(query):
    """Search arXiv and return paper metadata."""

    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": MAX_RESULTS_PER_QUERY,
        "sortBy": "relevance",
    })

    url = f"{ARXIV_API}?{params}"
    xml_data = fetch(url)

    root = ET.fromstring(xml_data)

    papers = []

    for entry in root.findall("atom:entry", NS):

        arxiv_id = entry.find("atom:id", NS).text.split("/abs/")[-1]

        title = entry.find("atom:title", NS).text.strip()
        title = title.replace("\n", " ")

        pdf_url = None

        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "pdf_url": pdf_url,
        })

    return papers


def download_pdf(paper):
    """Download one paper and return its local path."""

    filename = paper["arxiv_id"].replace("/", "_") + ".pdf"
    path = PAPERS_DIR / filename

    if not path.exists():
        data = fetch(paper["pdf_url"])

        with open(path, "wb") as f:
            f.write(data)

    return path


def main():

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    # Search arXiv
    papers = {}

    for query in SEARCH_QUERIES:

        print(f"Searching: {query}")

        try:
            results = search_arxiv(query)

            for paper in results:
                papers[paper["arxiv_id"]] = paper

            print(f"  found {len(results)} papers")

        except Exception as e:
            print(f"  search failed: {e}")

        time.sleep(SECONDS_BETWEEN_REQUESTS)

    print(f"\nFound {len(papers)} unique papers.")

    # Download PDFs
    metadata = []

    for paper in papers.values():

        try:
            path = download_pdf(paper)

            paper["local_path"] = str(path)
            metadata.append(paper)

            print(f"  downloaded: {paper['title'][:70]}")

        except Exception as e:
            print(f"  download failed: {paper['arxiv_id']}: {e}")

        time.sleep(SECONDS_BETWEEN_REQUESTS)

    # Save metadata
    metadata_path = PAPERS_DIR / "metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved {len(metadata)} papers to {metadata_path}")


if __name__ == "__main__":
    main()