"""
python scripts/ask.py  and you will be prompted to type a question. The script will retrieve relevant passages from the indexed corpus and synthesize an answer using a language model (if configured).

Backend for answer synthesis is picked via RAG_LLM_BACKEND env var:
  export RAG_LLM_BACKEND=ollama      # free, local, needs ollama running
  (unset -> falls back to just showing ranked passages, no LLM call at all)
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edmrag.retriever import TfidfRetriever
from edmrag import agent

INDEX_PATH = Path("data/index/tfidf_index.pkl")
DEFAULT_QUESTION = "what are major systematic errors in storage ring EDM searches?"


def main():
    if len(sys.argv) >= 2:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask a question: ").strip()
        if not question:
            question = DEFAULT_QUESTION
            print(f"(nothing typed, using default: {question!r})")

    if not INDEX_PATH.exists():
        print("no index found - run fetch_papers.py -> extract_text.py -> "
              "build_index.py first (or try demo_with_sample_corpus.py "
              "to see it work on sample data without fetching anything).")
        sys.exit(1)

    retriever = TfidfRetriever.load(INDEX_PATH)
    result = agent.ask(question, retriever)
    print(result["answer"])


if __name__ == "__main__":
    main()
