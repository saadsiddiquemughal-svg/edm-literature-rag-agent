# EDM Literature RAG Agent

A lightweight **Retrieval-Augmented Generation (RAG)** system for searching and answering questions from scientific literature related to **Electric Dipole Moment (EDM) experiments and storage rings**.

The project collects relevant papers from arXiv, extracts and chunks their text, builds a searchable TF-IDF index, retrieves relevant passages for a user query, and uses a local LLM through Ollama to generate a source-grounded answer.

## Architecture

```text
                    ┌──────────────┐
                    │    arXiv     │
                    │    Papers    │
                    └──────┬───────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ PDF Extraction  │
                  │  & Chunking     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   TF-IDF Index  │
                  └────────┬────────┘
                           │
                     User Query
                           │
                           ▼
                  ┌─────────────────┐
                  │    Retrieval    │
                  │ + Cosine Sim.   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Ollama / LLM   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Answer + Sources│
                  └─────────────────┘
```

## Features

* Search and download scientific papers from arXiv
* Extract text from PDF documents
* Create overlapping text chunks for retrieval
* TF-IDF and cosine similarity based retrieval
* Simple decomposition of compound questions
* Local LLM-based answer synthesis with Ollama
* Extractive fallback when LLM synthesis is unavailable
* Source attribution using arXiv IDs and paper titles
* Experimental sentence-transformer retrieval backend

## Project Structure

```text
edm-literature-rag-agent/
│
├── edmrag/
│   ├── agent.py          # RAG orchestration and answer synthesis
│   ├── retriever.py      # Retrieval implementations
│   └── __init__.py
│
├── scripts/
│   ├── fetch_papers.py   # Fetch papers from arXiv
│   ├── extract_text.py   # Extract and chunk PDF text
│   ├── build_index.py    # Build the TF-IDF index
│   └── ask.py            # Command-line question interface
│
├── data/
│   ├── papers/           # Downloaded papers and metadata
│   └── index/            # Generated chunks and retrieval index
│
├── requirements.txt
├── run_pipeline.sh
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/saadsiddiquemughal-svg/edm-literature-rag-agent.git
cd edm-literature-rag-agent
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Build the literature index

Run the complete pipeline:

```bash
./run_pipeline.sh
```

Or run the individual steps:

```bash
python scripts/fetch_papers.py
python scripts/extract_text.py
python scripts/build_index.py
```

### 5. Ask a question

```bash
python scripts/ask.py
```

Or provide a question directly:

```bash
python scripts/ask.py "How can EDM be measured in a storage ring?"
```

## LLM Backend

The project uses **Ollama** for local LLM-based answer generation.

Set the backend with:

```bash
export RAG_LLM_BACKEND=ollama
```

If LLM synthesis is unavailable, the system falls back to displaying the most relevant retrieved passages.

## Retrieval

The default retrieval method is:

**TF-IDF + cosine similarity**

An experimental `EmbeddingRetriever` using `sentence-transformers` is also included as a possible direction for semantic retrieval.

## Limitations

This is a lightweight research and learning project rather than a production-grade literature search system.

Current limitations include:

* TF-IDF relies primarily on lexical similarity
* Character-based chunking is relatively simple
* Query decomposition uses a basic rule-based approach
* No reranking stage is currently implemented
* LLM-generated answers should still be verified against the original papers

## Motivation

The project explores how a transparent RAG pipeline can be built for scientific literature while keeping the individual components simple enough to understand, inspect, and modify.

## License

See [LICENSE](LICENSE).
