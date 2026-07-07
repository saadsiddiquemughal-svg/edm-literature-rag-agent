"""
Retrieval backend. Default is TF-IDF + cosine similarity - no model
download needed, works offline, good enough to prove the pipeline works.

For real use I'd swap this for sentence-transformer embeddings (see
EmbeddingRetriever below) - better at matching a question phrased
differently than the paper's wording, which TF-IDF is bad at since it's
just word overlap. Left both in so it's a one-line swap depending on
whether you've got internet access to pull the model from huggingface.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = Path("data/index")


class TfidfRetriever:
    """Default retriever. No downloads, works anywhere."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), stop_words="english"
        )
        self.matrix = None
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)
        return self

    def query(self, question, top_k=5):
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            {**self.chunks[i], "score": float(sims[i])}
            for i in top_idx if sims[i] > 0
        ]

    def save(self, path=INDEX_DIR / "tfidf_index.pkl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer,
                        "matrix": self.matrix, "chunks": self.chunks}, f)

    @classmethod
    def load(cls, path=INDEX_DIR / "tfidf_index.pkl"):
        r = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        r.vectorizer, r.matrix, r.chunks = d["vectorizer"], d["matrix"], d["chunks"]
        return r


class EmbeddingRetriever:
    """
    Swap-in replacement using real sentence embeddings instead of TF-IDF.
    Needs `pip install sentence-transformers` and internet access to pull
    the model the first time - didn't have that available while building
    this, so it's untested here, but the interface matches TfidfRetriever
    so agent.py doesn't need to know or care which one it's talking to.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True,
                                            normalize_embeddings=True)
        return self

    def query(self, question, top_k=5):
        q_emb = self.model.encode([question], normalize_embeddings=True)[0]
        sims = self.embeddings @ q_emb
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            {**self.chunks[i], "score": float(sims[i])}
            for i in top_idx if sims[i] > 0
        ]

    def save(self, path=INDEX_DIR / "embedding_index.pkl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "chunks": self.chunks}, f)
