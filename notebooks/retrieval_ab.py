"""Compare retrieval modes on the spec test questions (no LLM calls).

    python notebooks/retrieval_ab.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.retrieval.retriever import search

QUESTIONS = [
    "What is AWS Lambda?",
    "How is Kinesis Data Streams different from Amazon Data Firehose?",
    "Which services are in the Analytics category?",
    "How much does Lambda cost per request?",
]

MODES = [
    ("dense", False, False),
    ("dense+bm25", True, False),
    ("dense+bm25+rerank", True, True),
]


def label(chunk) -> str:
    metadata = chunk.document.metadata
    start = metadata.get("printed_page_start") or metadata.get("page_start")
    return f"{metadata.get('section_path')} (p. {start})"


def main() -> None:
    for question in QUESTIONS:
        print("=" * 78)
        print(f"QUESTION: {question}")
        for name, use_bm25, use_reranker in MODES:
            config.USE_BM25 = use_bm25
            config.USE_RERANKER = use_reranker
            started = time.perf_counter()
            results = search(question, k=config.TOP_K)
            elapsed = time.perf_counter() - started
            print(f"\n  {name:<20} {elapsed * 1000:6.0f} ms")
            for rank, chunk in enumerate(results[:5], start=1):
                print(f"    {rank}. {chunk.score:.3f}  {label(chunk)}")


if __name__ == "__main__":
    main()
