# aws-rag

Question answering over the **Overview of Amazon Web Services** whitepaper
(162 pages, June 2026), with grounded answers and page citations.

Everything runs locally except the final answer: local BGE embeddings, a local
Chroma vector store, and Google Gemini Flash (free tier) for generation.

## Stack

| Layer | Choice |
|---|---|
| Loading | PyMuPDF (text + fonts), pypdf (PDF outline) |
| Splitting | One section per bookmark, `RecursiveCharacterTextSplitter` only for long sections |
| Embeddings | `BAAI/bge-base-en-v1.5`, normalized, local |
| Vector store | Chroma, cosine similarity, persisted in `data/chroma_db/` |
| Retrieval | **Hybrid (dense + BM25) + `BAAI/bge-reranker-base` cross-encoder** |
| Generation | Gemini Flash via `langchain-google-genai` |
| Interface | CLI |

## Setup

Requires Python 3.10+ (developed on 3.14).

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env and set GOOGLE_API_KEY=...
```

The first embedding run downloads `BAAI/bge-base-en-v1.5` (~440 MB) from
Hugging Face; enabling the reranker also pulls `BAAI/bge-reranker-base`
(~1.1 GB). After that the index can be built and searched offline.

## Run

```powershell
python scripts/01_ingest.py          # PDF -> data/processed/sections.json
python scripts/02_build_index.py     # chunks -> embeddings -> Chroma
python scripts/02_build_index.py --chunks-only   # chunk + stats only
python scripts/03_ask.py             # interactive CLI
python notebooks/retrieval_ab.py     # dense vs +BM25 vs +reranker (no LLM)
python evaluation/evaluate.py        # retrieval metrics (no LLM)
python evaluation/answer_eval.py     # citation + refusal checks (LLM)
```

`02_build_index.py` is idempotent: it drops and recreates the collection, so
re-running never duplicates documents.

## Configuration

All settings live in [`src/config.py`](src/config.py):

- `PDF_PATH`, `SECTIONS_PATH`, `CHUNKS_PATH`, `CHROMA_DIR`
- `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`, `QUERY_PREFIX`
- `CHUNK_SIZE` (1200), `CHUNK_OVERLAP` (150)
- `TOP_K` (5)
- `USE_BM25` (on), `USE_RERANKER` (on)
- `BM25_K`, `BM25_K1`, `BM25_B`, `BM25_WEIGHT`, `DENSE_WEIGHT`
- `USE_MMR`, `MMR_FETCH_K`, `MMR_LAMBDA`
- `RERANKER_MODEL`, `RERANK_FETCH_K`, `RERANK_TOP_K`
- `EXPAND_PARENT_SECTIONS`, `MAX_CONTEXT_CHUNKS`
- `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_OUTPUT_TOKENS`, `LLM_TIMEOUT`,
  `LLM_MAX_RETRIES`, `LLM_SEED`

### Retrieval pipeline

1. **Dense** — cosine search over Chroma, BGE query instruction on queries only.
2. **BM25** (`USE_BM25`, on by default) — lexical index over `chunks.json`,
   fused with the dense ranking by Reciprocal Rank Fusion (`DENSE_WEIGHT` 0.6 /
   `BM25_WEIGHT` 0.4, `RRF_K` 60). The fused `score` is normalized to the best
   candidate in the result set.
3. **MMR** (`USE_MMR`, off by default) — maximal marginal relevance re-selects
   the dense candidate pool (`MMR_FETCH_K` 20, `MMR_LAMBDA` 0.6) to trade a
   little relevance for section diversity before reranking. It takes precedence
   over BM25 when both are enabled.
4. **Rerank** (`USE_RERANKER`, on) — the top `RERANK_FETCH_K` (10) candidates
   are re-scored with the `BAAI/bge-reranker-base` cross-encoder and trimmed to
   `RERANK_TOP_K` (5). Scores are sigmoid probabilities from the cross-encoder.
5. **Context expansion** (`EXPAND_PARENT_SECTIONS`) — when only subsections are
   retrieved, the parent section's intro is appended, so a question like "what
   is SageMaker AI?" still gets the definition. The ranked top-k is unchanged;
   expanded chunks are labelled as such in the CLI.

Turn any layer off in `src/config.py` to fall back to a simpler pipeline.
`notebooks/retrieval_ab.py` prints top-5 results and timings side by side.

On the four test questions the reranker ranks the correct section first every
time, and it separates answerable from unanswerable questions: relevant chunks
score 0.89–0.99, while the best match for "How much does Lambda cost per
request?" scores 0.003. Rerank scores are probabilities, not cosine similarity,
and reranking adds roughly 5–6 s per question on CPU.

The default LLM is `gemini-3.5-flash-lite`, the free-tier Flash model
recommended for new projects in Google AI Studio (November 2026).

## Project layout

```
data/raw/            source PDF (gitignored)
data/processed/      sections.json and chunks.json (inspectable)
data/chroma_db/      persisted vector store (gitignored)
src/loading/         PDF extraction, cleaning, sectioning, table repair
src/chunking/        section-aware chunker
src/indexing/        BGE embeddings + Chroma builder
src/retrieval/       dense + BM25 hybrid + cross-encoder reranker
src/generation/      grounded prompt + Gemini chain
src/app/cli.py       interactive CLI
scripts/             01_ingest, 02_build_index, 03_ask
notebooks/           scratch inspection scripts
evaluation/          golden questions, retrieval + answer checks
```

## Page numbers

Citations use the **printed page number** in the document footer, which is the
number a reader sees in a PDF viewer's page field after the front matter
(PDF page 49 = printed page 39). Both are stored in section and chunk
metadata (`page_start` / `printed_page_start`).

## Known environment issue

This machine's Windows Application Control policy blocks the compiled
`_xxhash` extension. `langchain-core` imports `langsmith`, which imports
`xxhash` eagerly even though it is only used for deterministic trace IDs.
`src/compat.py` installs a deterministic BLAKE2b stand-in **only when the real
module cannot be imported**, and prints a warning. Nothing in this project uses
LangSmith tracing, so answers are unaffected.

## Evaluation

`evaluation/evaluate.py` runs a 16-question golden set (13 answerable, 3
unanswerable) through every retrieval mode, offline and without an API key.
Metrics are section-level Hit@k, Recall@k, MRR and nDCG@k.

| Mode | Hit@5 | Recall@5 | MRR | nDCG@5 | ms/query |
|---|---|---|---|---|---|
| dense | 1.000 | 1.000 | 0.962 | 0.962 | 82 |
| + BM25 | 1.000 | 1.000 | 0.885 | 0.916 | 79 |
| + rerank | 1.000 | 1.000 | **1.000** | **0.994** | 6813 |
| + BM25 + rerank | 1.000 | 1.000 | **1.000** | **0.994** | 7210 |

Dense retrieval already finds the right section for every question; the
reranker fixes the remaining ranking imperfections. Two tuning findings:

- **BM25 is off by default.** Alone it lowers MRR (0.962 → 0.885) because
  service-name queries match category and comparison sections; with the
  reranker it changes nothing measurable. It stays implemented for rare exact
  terms — set `USE_BM25 = True` to enable it.
- **`RERANK_FETCH_K` drives latency, not quality.** Halving the candidate pool
  from 20 to 10 keeps MRR 1.000 and nDCG 0.994 while cutting reranking from
  6.8 s to **2.8 s per question**; 5 candidates gives 1.6 s but leaves no
  margin for recovering a miss. Default is 10.

### Answer-level checks

`evaluation/answer_eval.py` calls the real chain and validates every citation
against the retrieved chunks: the cited page must overlap a retrieved chunk and
the cited label must match its section. It also checks that the three
unanswerable questions are refused and that no answerable one is.

```powershell
python evaluation/answer_eval.py --show-answers
```

Results over all 16 questions: **55 citations, 0 invalid** (precision 1.000),
13/13 answerable questions cited, and 3/3 unanswerable ones refused. This
check caught a real bug: "What is Amazon SageMaker AI?" was sometimes refused
because reranking returned only child sections (Pipelines, Feature Store, ...)
and never the parent overview. Parent-section expansion fixed it.
