"""Cross-encoder reranking for retrieved chunks.

The reranker is optional (``config.USE_RERANKER``). It re-scores a larger
candidate set with a cross-encoder and keeps the best ``RERANK_TOP_K`` chunks,
which is much more precise than cosine similarity alone but slower.

``CrossEncoder.predict`` already applies a sigmoid for single-label models, so
its output is used directly as a 0..1 relevance probability.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

import src.config as config
from src.indexing.embeddings import resolve_device
from src.retrieval.types import RetrievedChunk


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load the configured cross-encoder once per process."""
    return CrossEncoder(config.RERANKER_MODEL, device=resolve_device())


def _body(chunk: RetrievedChunk) -> str:
    """Rerank on the chunk body; the repeated section header adds no signal."""
    return chunk.document.page_content.split("\n\n", 1)[-1].strip()


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Return the top_k chunks ordered by cross-encoder relevance."""
    if not chunks:
        return []

    model = get_reranker()
    scores = model.predict(
        [(query, _body(chunk)) for chunk in chunks], show_progress_bar=False
    )
    scored = [
        RetrievedChunk(document=chunk.document, score=float(score))
        for chunk, score in zip(chunks, scores, strict=True)
    ]
    scored.sort(key=lambda chunk: chunk.score, reverse=True)
    return scored[:top_k]
