"""Central configuration for the AWS whitepaper RAG project.

Every path, model name and tunable lives here so nothing is hardcoded in the
pipeline modules.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CHROMA_DIR: Path = DATA_DIR / "chroma_db"

PDF_PATH: Path = RAW_DIR / "aws-overview.pdf"
SECTIONS_PATH: Path = PROCESSED_DIR / "sections.json"
CHUNKS_PATH: Path = PROCESSED_DIR / "chunks.json"

# Folder that holds the PDF bookmarks used as the section hierarchy.
OUTLINE_TOC_TITLE: str = "Table of Contents"

# Short prefix prepended to every chunk's embedded text.
SECTION_HEADER_PREFIX: str = "AWS Overview"

# ---------------------------------------------------------------------------
# Loading / cleaning
# ---------------------------------------------------------------------------
# Running header found at the top of every page.
PAGE_HEADER: str = "Overview of Amazon Web Services"
PAGE_HEADER_SUFFIX: str = "AWS Whitepaper"

# Font used for headings in the PDF (fallback heading detection).
HEADING_FONT: str = "AmazonEmber-Bold"

# A dot-leader line looks like: "Amazon EC2 .......... 39"
DOT_LEADER_MIN_DOTS: int = 5

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 1200         # characters (~300 tokens)
CHUNK_OVERLAP: int = 150       # characters (~12%)
MIN_CHUNK_CHARS: int = 40      # drop trailing scraps shorter than this

# ---------------------------------------------------------------------------
# Embeddings / vector store
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
EMBEDDING_DEVICE: str = "auto"  # "auto" -> cuda if available, else cpu
NORMALIZE_EMBEDDINGS: bool = True

# BGE retrieval instruction: queries only, never documents.
QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

CHROMA_COLLECTION: str = "aws_overview"
CHROMA_DISTANCE: str = "cosine"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K: int = 5

# Optional layers. Both are implemented; flags control whether they run.
# BM25 is on by default when used with reranker (see evaluation/README.md).
USE_BM25: bool = True
USE_RERANKER: bool = True

# MMR (maximal marginal relevance): diversify the dense candidate pool before
# reranking. Off by default; measured in evaluation/evaluate.py.
USE_MMR: bool = False
MMR_FETCH_K: int = 20     # dense candidates considered before selection
MMR_LAMBDA: float = 0.6   # 1.0 = pure relevance, 0.0 = pure diversity

# BM25 lexical scoring and dense/lexical fusion weights.
BM25_K: int = 10          # lexical candidates fetched before fusion
BM25_K1: float = 1.5      # term-frequency saturation
BM25_B: float = 0.75      # document-length normalization
BM25_WEIGHT: float = 0.4
DENSE_WEIGHT: float = 0.6
RRF_K: int = 60           # reciprocal rank fusion constant

RERANKER_MODEL: str = "BAAI/bge-reranker-base"
RERANK_FETCH_K: int = 10   # candidates fetched before reranking (2x top_k)
RERANK_TOP_K: int = 5      # kept after reranking

# De-duplication: merge all retrieved chunks of the same section into one
# context block. Long sections are split across chunks, so without this the
# same section can take two of the top-k slots and repeat text in the answer.
MERGE_SECTIONS: bool = True

# Context expansion: add the parent section intro when only subsections were
# retrieved (e.g. "SageMaker AI > Pipelines" without the SageMaker overview).
EXPAND_PARENT_SECTIONS: bool = True
MAX_CONTEXT_CHUNKS: int = 8

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
# Free-tier Flash model as listed in Google AI Studio (Nov 2026). The docs
# recommend Gemini 3.5 Flash-Lite for new projects and route 2.5 Flash to it.
LLM_MODEL: str = "gemini-3.5-flash-lite"
LLM_TEMPERATURE: float = 0.1
LLM_MAX_OUTPUT_TOKENS: int = 1024
LLM_TIMEOUT: float = 45.0   # seconds per request; a stalled call fails fast
LLM_MAX_RETRIES: int = 1    # SDK-level retries on 429/5xx before our own
LLM_SEED: int = 7           # helps keep free-tier sampling consistent
