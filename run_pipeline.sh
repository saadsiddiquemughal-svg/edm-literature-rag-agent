#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -f data/papers/metadata.json ]; then
    echo "no papers found yet, fetching from arXiv (takes a few minutes)..."
    python scripts/fetch_papers.py
else
    echo "papers already fetched, skipping (delete data/papers/metadata.json to re-fetch)"
fi

if [ ! -f data/index/chunks.json ]; then
    echo "extracting text from PDFs..."
    python scripts/extract_text.py
else
    echo "chunks already extracted, skipping"
fi

if [ ! -f data/index/tfidf_index.pkl ]; then
    echo "building index..."
    python scripts/build_index.py
else
    echo "index already built, skipping"
fi

echo ""
echo "ready - ask a question"
python scripts/ask.py