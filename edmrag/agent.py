"""
Small hand-rolled agent, three steps:

  1. plan()      - break a question into sub-questions if it's a compound one
  2. retrieve()  - run each sub-question through the retriever, merge & dedupe
  3. synthesize() - turn the retrieved chunks into an answer with citations

No agent framework here on purpose - for a 3-step pipeline, LangChain-type
abstractions add more boilerplate than they save, and rolling it by hand
means I actually understand every step instead of trusting a framework's
defaults.

Synthesis has two modes:
  - LLM mode:  use ollama LLM to write the answer
    from the retrieved passages, citing paper titles/arXiv IDs.
  - Extractive fallback (default, no API key needed): just returns the
    top-ranked passages grouped by paper, so you get citations without
    needing to pay for API calls. Good enough to sanity-check the
    retrieval quality, which is honestly the harder part of RAG anyway.
"""

import os
import re
import urllib.request
import json as _json


def plan(question):
    """Very simple decomposition - split on ' and ' if it looks like a
    compound question. Not trying to be clever here, just handles the
    common case of someone stacking two questions together."""
    parts = re.split(r"\band\b", question, flags=re.IGNORECASE)
    parts = [p.strip(" ?.") for p in parts if len(p.strip()) > 15]
    return parts if len(parts) > 1 else [question]


def retrieve(sub_questions, retriever, top_k=4):
    seen = set()
    merged = []
    for sq in sub_questions:
        for hit in retriever.query(sq, top_k=top_k):
            key = (hit["arxiv_id"], hit["text"][:80])
            if key not in seen:
                seen.add(key)
                merged.append(hit)
    merged.sort(key=lambda h: h["score"], reverse=True)
    return merged


def extractive_answer(question, hits):
    if not hits:
        return "No relevant passages found in the indexed papers for this question."
    by_paper = {}
    for h in hits:
        by_paper.setdefault((h["arxiv_id"], h["title"]), []).append(h)

    lines = [f"Found {len(hits)} relevant passage(s) across {len(by_paper)} paper(s):\n"]
    for (arxiv_id, title), paper_hits in by_paper.items():
        lines.append(f"[{arxiv_id}] {title}")
        for h in paper_hits[:2]:  
            snippet = h["text"][:300].strip()
            lines.append(f"  \u2192 \"{snippet}...\"  (score {h['score']:.2f})")
        lines.append("")
    return "\n".join(lines)




def ollama_answer(question, hits, model="deepseek-r1:1.5b"):
    """ollama answer synthesis backend - local, free, needs ollama running.
    If you want to use a different LLM backend, just implement a function"""

    context = "\n\n".join(
        f"[Source: {h['arxiv_id']} - {h['title']}]\n{h['text']}" for h in hits
    )
    prompt = (
        "Answer the question using only the sources below. Cite the arXiv ID "
        "in brackets after each claim, like [2103.01234]. If the sources don't "
        "cover the question, say so plainly instead of guessing.\n\n"
        f"Sources:\n{context}\n\nQuestion: {question}"
    )
    payload = _json.dumps({
        "model": model, "prompt": prompt, "stream": False
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return _json.loads(resp.read())["response"]


def synthesize(question, hits, backend=None):
    """backend is one of: 'ollama(default) and 'extractive'.
    Set via the RAG_LLM_BACKEND env var, or pass it directly. Extractive
    needs nothing installed and costs nothing, which is why it's the
    fallback if a backend errors out rather than just crashing."""
    backend = backend or os.environ.get("RAG_LLM_BACKEND", "ollama").lower()
    try:
        if backend == "ollama":
            return ollama_answer(question, hits)
    except Exception as e:
        return f"({backend} synthesis failed: {e})\n\n" + extractive_answer(question, hits)
    return extractive_answer(question, hits)

def format_sources(hits):
    seen = {}
    for h in hits:
        seen.setdefault(h["arxiv_id"], h["title"])
    lines = ["\nSources used:"]
    for arxiv_id, title in seen.items():
        lines.append(f"  [{arxiv_id}] {title}")
    return "\n".join(lines)

def ask(question, retriever, top_k=4, backend=None):
    sub_qs = plan(question)
    hits = retrieve(sub_qs, retriever, top_k=top_k)
    answer = synthesize(question, hits, backend=backend)
    if hits:
        answer = answer.rstrip() + "\n\n" + format_sources(hits)
    return {
        "question": question,
        "sub_questions": sub_qs,
        "sources": [{"arxiv_id": h["arxiv_id"], "title": h["title"], "score": h["score"]}
                   for h in hits],
        "answer": answer,
    }
