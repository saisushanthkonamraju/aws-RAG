"""Lexical BM25 index over the processed chunks.

A small, dependency-free Okapi BM25 implementation. It is used only for the
optional hybrid retriever (``config.USE_BM25``) and never replaces dense search
on its own.

The index is built from ``data/processed/chunks.json``, so it always mirrors
exactly what was embedded into Chroma.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document

import src.config as config

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words carry no retrieval signal and only dilute BM25 scores.
_STOPWORDS = frozenset(
    """a an the of in on for to and or is are was were be been being what which
    how does do did with by at from that this it its as but not no can will
    would should could i you we they he she them their there here about into
    over under between per than then so if when where who whom why""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; hyphens and punctuation split, versions survive."""
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


@dataclass
class _Entry:
    """One indexed chunk: its document and precomputed term frequencies."""

    document: Document
    term_freqs: Counter[str]
    length: int


class BM25Index:
    """In-memory Okapi BM25 index with the standard k1/b parameters."""

    def __init__(self, entries: list[_Entry], k1: float, b: float) -> None:
        self._entries = entries
        self._k1 = k1
        self._b = b
        self._doc_count = len(entries)
        self._avg_length = (
            sum(entry.length for entry in entries) / self._doc_count
            if self._doc_count
            else 0.0
        )

        doc_freqs: Counter[str] = Counter()
        for entry in entries:
            doc_freqs.update(entry.term_freqs.keys())

        self._idf = {
            term: math.log(1 + (self._doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freqs.items()
        }

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Return the k highest-scoring documents for a query."""
        terms = tokenize(query)
        if not terms or not self._entries:
            return []

        scored: list[tuple[Document, float]] = []
        for entry in self._entries:
            score = 0.0
            norm = self._k1 * (
                1 - self._b + self._b * entry.length / (self._avg_length or 1.0)
            )
            for term in terms:
                freq = entry.term_freqs.get(term)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                score += idf * freq * (self._k1 + 1) / (freq + norm)
            if score > 0:
                scored.append((entry.document, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


def _entries_from_chunks(chunks: list[dict]) -> list[_Entry]:
    entries: list[_Entry] = []
    for chunk in chunks:
        text = chunk["text"]
        tokens = tokenize(text)
        entries.append(
            _Entry(
                document=Document(page_content=text, metadata=chunk["metadata"]),
                term_freqs=Counter(tokens),
                length=len(tokens),
            )
        )
    return entries


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    """Build (once per process) the index over the saved chunks."""
    if not config.CHUNKS_PATH.exists():
        raise SystemExit(
            f"{config.CHUNKS_PATH} not found. Run scripts/02_build_index.py --chunks-only first."
        )
    chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
    return BM25Index(_entries_from_chunks(chunks), config.BM25_K1, config.BM25_B)
