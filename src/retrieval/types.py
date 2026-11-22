"""Shared retrieval types (kept separate to avoid import cycles)."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class RetrievedChunk:
    """A retrieved document with its relevance score (higher is better)."""

    document: Document
    score: float
    origin: str = "search"  # "search" or "parent" (context expansion)
