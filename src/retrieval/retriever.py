"""Retrieval: dense search, optional BM25 hybrid, optional reranking.

The pipeline is built from three independent layers, each behind a flag in
``src/config.py``:

* dense  (always)      cosine similarity over the Chroma collection
* BM25   (USE_BM25)    lexical ranking fused with dense via reciprocal rank
* rerank (USE_RERANKER) cross-encoder over a larger candidate pool
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document

import src.config as config
from src.indexing.build_index import load_vector_store
from src.indexing.embeddings import get_embeddings
from src.retrieval.bm25 import get_bm25_index
from src.retrieval.types import RetrievedChunk


@lru_cache(maxsize=1)
def get_store():
    """Open the persisted vector store once per process."""
    return load_vector_store()


def _dense_search(store, query: str, k: int) -> list[RetrievedChunk]:
    """Cosine similarity over the Chroma collection (higher = closer)."""
    results = store.similarity_search_with_score(f"{config.QUERY_PREFIX}{query}", k=k)
    return [
        RetrievedChunk(document=document, score=1.0 - float(distance))
        for document, distance in results
    ]


def _bm25_search(query: str, k: int) -> list[RetrievedChunk]:
    """Lexical BM25 matches, useful for exact names and rare terms."""
    return [
        RetrievedChunk(document=document, score=score)
        for document, score in get_bm25_index().search(query, k)
    ]


def _key(chunk: RetrievedChunk) -> tuple[str, int]:
    """Identity of a chunk across both retrievers (ids live in metadata)."""
    metadata = chunk.document.metadata
    return (str(metadata.get("section_id")), int(metadata.get("chunk_index", 0)))


def _hybrid_search(store, query: str, k: int) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion of the dense and lexical rankings.

    RRF combines ranks rather than raw scores, so an unbounded BM25 score and a
    0..1 cosine score never have to be forced onto the same scale. The reported
    score is normalized to the best candidate (1.0 = top of this result set),
    not a calibrated probability.
    """
    candidate_k = max(k, config.BM25_K)
    dense = _dense_search(store, query, candidate_k)
    lexical = _bm25_search(query, candidate_k)

    documents: dict[tuple[str, int], Document] = {}
    fused: dict[tuple[str, int], float] = {}

    for weight, ranking in (
        (config.DENSE_WEIGHT, dense),
        (config.BM25_WEIGHT, lexical),
    ):
        for rank, chunk in enumerate(ranking):
            key = _key(chunk)
            documents[key] = chunk.document
            fused[key] = fused.get(key, 0.0) + weight / (config.RRF_K + rank + 1)

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
    if not ranked:
        return []
    best = ranked[0][1]
    return [
        RetrievedChunk(document=documents[key], score=score / best)
        for key, score in ranked
    ]


@lru_cache(maxsize=1)
def _parent_intros() -> dict[str, Document]:
    """First chunk of every section, for parent-section context expansion."""
    chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
    intros: dict[str, Document] = {}
    for chunk in chunks:
        metadata = chunk["metadata"]
        if int(chunk["metadata"].get("chunk_index", 0)) == 0:
            intros.setdefault(
                metadata["section_path"],
                Document(page_content=chunk["text"], metadata=metadata),
            )
    return intros


def _expand_parents(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Append the parent section intro when only its subsections were retrieved.

    A child section such as "Amazon SageMaker AI > Pipelines" does not define
    the service; the parent section does. Adding it keeps answers grounded in
    the overview text without replacing the ranked results.
    """
    if not config.EXPAND_PARENT_SECTIONS or not chunks:
        return chunks

    intros = _parent_intros()
    selected = {_key(chunk) for chunk in chunks}
    expanded = list(chunks)

    for chunk in chunks:
        if len(expanded) >= config.MAX_CONTEXT_CHUNKS:
            break
        path = str(chunk.document.metadata.get("section_path", ""))
        if " > " not in path:
            continue
        parent_path = path.rsplit(" > ", 1)[0]
        # Skip top-level categories ("Compute", "Analytics", ...): their intro
        # is generic boilerplate, not context for a specific service.
        if " > " not in parent_path:
            continue
        parent = intros.get(parent_path)
        if parent is None:
            continue
        key = (str(parent.metadata.get("section_id")), int(parent.metadata.get("chunk_index", 0)))
        if key in selected:
            continue
        selected.add(key)
        expanded.append(
            RetrievedChunk(document=parent, score=chunk.score, origin="parent")
        )
    return expanded


@lru_cache(maxsize=1)
def _vector_lookup() -> dict[str, np.ndarray]:
    """Chunk text -> embedding, for local MMR re-selection."""
    data = get_store().get(include=["embeddings", "documents"])
    documents = data.get("documents") or []
    embeddings = data.get("embeddings")
    if embeddings is None:
        return {}
    return {
        text: np.asarray(vector, dtype=float)
        for text, vector in zip(documents, embeddings, strict=False)
    }


def _mmr_select(
    query: str, candidates: list[RetrievedChunk], k: int, lambda_mult: float
) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance over an already retrieved candidate pool."""
    if len(candidates) <= k:
        return candidates

    vectors = _vector_lookup()
    query_vector = np.asarray(
        get_embeddings().embed_query(f"{config.QUERY_PREFIX}{query}"), dtype=float
    )
    matrix = np.asarray(
        [
            vectors.get(chunk.document.page_content, np.zeros_like(query_vector))
            for chunk in candidates
        ]
    )
    relevance = matrix @ query_vector

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda index: relevance[index])
        else:
            redundancy = (matrix[remaining] @ matrix[selected].T).max(axis=1)
            combined = lambda_mult * relevance[remaining] - (1 - lambda_mult) * redundancy
            best = remaining[int(np.argmax(combined))]
        selected.append(best)
        remaining.remove(best)

    return [
        RetrievedChunk(document=candidates[index].document, score=float(relevance[index]))
        for index in selected
    ]


def _merge_bodies(left: str, right: str) -> str:
    """Join two chunk bodies, dropping the overlap the chunker repeated."""
    limit = min(len(left), len(right), 400)
    for size in range(limit, 39, -1):
        if left.endswith(right[:size]):
            return left + right[size:]
    return f"{left}\n\n{right}"


def _merge_sections(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Merge chunks of the same section into one context block.

    Long sections are split by the chunker, so two chunks of one section can
    take two of the five slots and show the model the same text twice. Merging
    keeps every chunk's content (including a list that spans a split), removes
    the repeated section header and overlap, and restores the section's full
    page span. The group inherits the best score of its members.
    """
    groups: dict[str, list[RetrievedChunk]] = {}
    order: list[str] = []
    for chunk in chunks:
        path = str(chunk.document.metadata.get("section_path", ""))
        if path not in groups:
            groups[path] = []
            order.append(path)
        groups[path].append(chunk)

    merged: list[RetrievedChunk] = []
    for path in order:
        members = groups[path]
        if len(members) == 1:
            merged.append(members[0])
            continue

        ordered = sorted(
            members,
            key=lambda chunk: int(chunk.document.metadata.get("chunk_index", 0)),
        )
        bodies = [
            chunk.document.page_content.split("\n\n", 1)[-1].strip()
            for chunk in ordered
        ]
        body = bodies[0]
        for extra in bodies[1:]:
            body = _merge_bodies(body, extra)

        best = members[0]  # members keep the ranking order
        metadata = dict(best.document.metadata)
        starts: list[int] = [
            int(c.document.metadata.get("printed_page_start", 0))  # type: ignore
            for c in members
            if c.document.metadata.get("printed_page_start") is not None
        ]
        ends: list[int] = [
            int(c.document.metadata.get("printed_page_end", 0))  # type: ignore
            for c in members
            if c.document.metadata.get("printed_page_end") is not None
        ]
        if starts:
            metadata["printed_page_start"] = min(starts)  # type: ignore
        if ends:
            metadata["printed_page_end"] = max(ends)  # type: ignore
        metadata["merged_chunks"] = len(members)

        merged.append(
            RetrievedChunk(
                document=Document(
                    page_content=f"{config.SECTION_HEADER_PREFIX} > {path}\n\n{body}",
                    metadata=metadata,
                ),
                score=best.score,
            )
        )
    return merged


def search(query: str, k: int | None = None, store=None) -> list[RetrievedChunk]:
    """Return the top-k chunks for a question, most relevant first."""
    if not query.strip():
        return []

    top_k = k or config.TOP_K
    vector_store = store or get_store()

    candidate_k = top_k
    if config.USE_RERANKER:
        candidate_k = max(top_k, config.RERANK_FETCH_K)
    elif config.MERGE_SECTIONS:
        # Without a reranker, fetch extra candidates so merging still fills k.
        candidate_k = max(top_k, top_k * 2)

    if config.USE_BM25:
        candidates = _hybrid_search(vector_store, query, candidate_k)
    elif config.USE_MMR:
        pool = _dense_search(
            vector_store, query, max(config.MMR_FETCH_K, candidate_k)
        )
        candidates = _mmr_select(query, pool, candidate_k, config.MMR_LAMBDA)
    else:
        candidates = _dense_search(vector_store, query, candidate_k)

    if config.USE_RERANKER:
        # Imported lazily so the cross-encoder is never loaded when unused.
        from src.retrieval.reranker import rerank

        scored = rerank(query, candidates, len(candidates))
    else:
        scored = candidates

    if config.MERGE_SECTIONS:
        scored = _merge_sections(scored)
    return _expand_parents(scored[:top_k])
