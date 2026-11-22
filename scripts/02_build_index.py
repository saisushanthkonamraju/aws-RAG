"""Stage 2/3 entry point: chunk the sections, then build the Chroma index.

Stage 2 stops after writing data/processed/chunks.json; Stage 3 adds embedding
and persistence below.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.chunking.chunker import Chunk, build_chunks
from src.indexing.build_index import build_index, collection_count
from src.indexing.embeddings import get_embeddings, resolve_device


def load_sections() -> list[dict]:
    if not config.SECTIONS_PATH.exists():
        raise SystemExit(f"{config.SECTIONS_PATH} not found. Run scripts/01_ingest.py first.")
    return json.loads(config.SECTIONS_PATH.read_text(encoding="utf-8"))


def print_statistics(chunks: list[Chunk]) -> None:
    lengths = [len(chunk.text) for chunk in chunks]
    split_sections = Counter(chunk.metadata["section_id"] for chunk in chunks)

    print(f"\nChunks: {len(chunks)} from {len(split_sections)} sections")
    print(f"Sections split into multiple chunks: {sum(1 for n in split_sections.values() if n > 1)}")
    print(
        "Length (chars incl. header): "
        f"min {min(lengths)} | median {int(statistics.median(lengths))} | "
        f"mean {int(statistics.mean(lengths))} | max {max(lengths)}"
    )

    by_type = Counter(chunk.metadata["doc_type"] for chunk in chunks)
    print("By doc_type: " + ", ".join(f"{name} {count}" for name, count in by_type.most_common()))

    buckets = [0, 300, 600, 900, 1200, 1500, 10_000]
    print("Length distribution:")
    for low, high in zip(buckets, buckets[1:]):
        count = sum(1 for length in lengths if low <= length < high)
        print(f"  {low:5d}-{high:5d}: {'#' * round(count / 2):<40} {count}")


def print_samples(chunks: list[Chunk]) -> None:
    wanted = [
        ("service", 0),
        ("category_intro", 0),
        ("comparison_table", 0),
        ("concept", 0),
    ]
    shown: list[Chunk] = []
    for doc_type, index in wanted:
        match = next(
            (
                chunk
                for chunk in chunks
                if chunk.metadata["doc_type"] == doc_type and chunk.metadata["chunk_index"] == index
            ),
            None,
        )
        if match:
            shown.append(match)

    split = next(
        chunk
        for chunk in chunks
        if chunk.metadata["chunk_index"] == 1
    )
    shown.append(split)

    for chunk in shown:
        print("\n" + "=" * 78)
        print(f"{chunk.id}  [{chunk.metadata['doc_type']}]  chars={len(chunk.text)}")
        print("=" * 78)
        print(chunk.text[:700] + ("..." if len(chunk.text) > 700 else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Only chunk and save chunks.json; skip embedding and indexing.",
    )
    args = parser.parse_args()

    sections = load_sections()
    chunks = build_chunks(sections)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_PATH.write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_statistics(chunks)
    print_samples(chunks)
    print(f"\nWrote {len(chunks)} chunks to {config.CHUNKS_PATH}")

    if args.chunks_only:
        return

    print(f"\nLoading {config.EMBEDDING_MODEL} on {resolve_device()} ...")
    get_embeddings()
    store = build_index([chunk.to_dict() for chunk in chunks])
    print(f"Indexed {collection_count(store)} chunks in {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
