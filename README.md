# Literature RAG agent

A small retrieval-augmented question-answering system for scientific
literature. Point it at a research area with a handful of search terms,
it pulls the matching papers from arXiv, indexes them, and answers
questions with citations back to the specific paper each answer came
from - so you can go check the source instead of just trusting it.

Built and tested on EDM (electric dipole moment) storage-ring physics -
my own PhD field - but nothing in the pipeline is EDM-specific. Swap the
search terms and it works for any field with a presence on arXiv.

## How it's built

Three stages get you from "a field of research" to "a searchable index":

1. **fetch_papers.py** - queries the arXiv API with a list of search
   terms, downloads matching PDFs + metadata.
2. **extract_text.py** - pulls text out of the PDFs (pdfplumber), splits
   it into overlapping chunks.
3. **build_index.py** - builds a TF-IDF index over the chunks.

Then **agent.py** answers questions in three steps:

- `plan()` - splits a compound question into sub-questions if needed
- `retrieve()` - runs each sub-question against the index, merges and
  dedupes the results
- `synthesize()` - turns the retrieved passages into an answer

No agent framework - for three steps, writing it by hand is less code
than learning someone else's abstraction, and it means every step is
something I can actually explain rather than trusting a library's
defaults.

Synthesis runs on a local Ollama model by default (free, no API key,
runs on your own machine). If Ollama isn't running, it falls back to
just listing the top-ranked passages grouped by paper instead of
crashing. Either way, every answer ends with a **Sources used** list -
built directly from what was actually retrieved, not from the model
remembering to cite properly on its own.

## Using it for a different field

`SEARCH_QUERIES` in `fetch_papers.py` is just a plain list of search
phrases sent to arXiv - nothing downstream assumes anything about the
topic. Swap the list, rerun the pipeline, and you have a literature
agent for a different field. A few examples of what precise queries
look like for other areas:

| Field | Example search queries |
|---|---|
| EDM storage-ring physics (built-in) | `"electric dipole moment of charged particles in storage rings"`, `"frozen spin technique for protons and deuterons in a storage ring"`, `"COSY spin coherence time of proton and deuterons in a storage ring"`, `"systematic errors in EDM storage ring"`, `"axion particles in a storage ring"`, `"beam dynamics simulation in EDM storage ring"` |
| Quantum error correction | `"surface code quantum error correction"`, `"logical qubit fault tolerant threshold"`, `"quantum LDPC codes"` |
| Battery / energy storage materials | `"solid state electrolyte lithium battery"`, `"dendrite suppression lithium metal anode"`, `"cathode degradation mechanism"` |
| Exoplanet detection | `"transit photometry exoplanet detection"`, `"radial velocity method exoplanet"`, `"atmospheric biosignature spectroscopy"` |
| Protein structure prediction | `"AlphaFold protein structure prediction accuracy"`, `"protein folding molecular dynamics simulation"` |
| Climate model uncertainty | `"climate model parameterization uncertainty"`, `"CMIP6 model intercomparison"`, `"cloud feedback climate sensitivity"` |

The pattern that makes a good query set: 2-4 word noun phrases that are
specific enough to avoid unrelated matches (a query like just `"rings"`
or `"battery"` alone pulls in completely unrelated papers - pure math
"rings" or networked control systems that happen to use the word
"battery"), but not so narrow that you only get one or two results per
query. Combine several angles on the same field (a core technique, a
key open problem, a specific measurement/method) rather than one broad
term, the same way the EDM list mixes the experimental method, the
facility, and the systematic-error angle.

## Retrieval backend

Default is TF-IDF + cosine similarity - no model download needed, works
anywhere, good baseline. `retriever.py` also has `EmbeddingRetriever`,
same interface, using sentence-transformer embeddings instead - better
at matching a question phrased differently than the paper's own
wording, since TF-IDF is really just word overlap. Swapping which one
`build_index.py` and `ask.py` use is a one-line change.

## No paper content is committed to this repo

Only the code to fetch and process papers yourself is here - not the
PDFs, not the extracted text, not the built index. `data/papers` and
`data/index` are gitignored, so anyone using this rebuilds their own
corpus for whatever field they point it at.

## Running it

```bash
pip install -r requirements.txt
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This fetches papers, extracts text, and builds the index - skipping any
stage that's already done - then drops you straight into asking a
question.

Or run each stage yourself:

```bash
python scripts/fetch_papers.py     # needs internet access to arxiv.org
python scripts/extract_text.py
python scripts/build_index.py
python scripts/ask.py "What causes geometric-phase systematic errors in EDM storage rings?"
```

For Ollama answers, `ollama serve` needs to be running with a model
pulled (`ollama pull deepseek-r1:1.5b` is a reasonable small default).

## Limitations

- TF-IDF retrieval is word-overlap based, not semantic - it can miss a
  passage that answers the question using different vocabulary than the
  question itself. The `EmbeddingRetriever` swap addresses this.
- The query planner only splits on "and" - a real implementation would
  use an LLM call to decompose compound questions properly.
- No re-ranking step after retrieval - a cross-encoder re-ranker on the
  top ~20 candidates would likely improve precision over raw
  TF-IDF/embedding similarity scores.
- Chunking is fixed-size by character count, which sometimes cuts a
  sentence in half; chunking by paragraph/section boundary would be
  cleaner.
- Small local LLMs (the kind that run on a laptop via Ollama) can
  overstate what the retrieved sources actually support, rather than
  sticking strictly to them - worth treating any generated answer as a
  starting point to verify against the listed sources, not a final word.

## Me

Saad Siddique - PhD candidate in accelerator physics, RWTH Aachen.
Thesis on beam dynamics simulations for an EDM storage ring, done in
collaboration with FZ Jülich, GSI, and CERN.
