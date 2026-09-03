import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edmrag.retriever import TfidfRetriever

CHUNKS_PATH = Path("data/index/chunks.json")


def main():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    print(f"building index from {len(chunks)} chunks...")
    retriever = TfidfRetriever().build(chunks)
    retriever.save()
    print("saved to data/index/tfidf_index.pkl")


if __name__ == "__main__":
    main()
