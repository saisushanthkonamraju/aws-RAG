"""Embed chunks with BGE and persist them in a local Chroma collection."""

from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_chroma import Chroma

import src.config as config
from src.indexing.embeddings import get_embeddings

BATCH_SIZE = 64


def _client(chroma_dir: Path) -> chromadb.ClientAPI:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def load_vector_store(chroma_dir: Path = config.CHROMA_DIR) -> Chroma:
    """Open the persisted collection for retrieval."""
    client = _client(chroma_dir)
    if config.CHROMA_COLLECTION not in {item.name for item in client.list_collections()}:
        raise SystemExit(
            f"Collection {config.CHROMA_COLLECTION!r} not found. "
            "Run scripts/02_build_index.py first."
        )
    return Chroma(
        client=client,
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": config.CHROMA_DISTANCE},
    )


def build_index(chunks: list[dict], chroma_dir: Path = config.CHROMA_DIR) -> Chroma:
    """Rebuild the collection from scratch (idempotent, no duplicates)."""
    client = _client(chroma_dir)
    existing = {item.name for item in client.list_collections()}
    if config.CHROMA_COLLECTION in existing:
        print(f"Dropping existing collection {config.CHROMA_COLLECTION!r}")
        client.delete_collection(config.CHROMA_COLLECTION)

    store = Chroma(
        client=client,
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": config.CHROMA_DISTANCE},
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        store.add_texts(
            texts=[chunk["text"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
            ids=[chunk["id"] for chunk in batch],
        )
        print(f"  embedded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    return store


def collection_count(store: Chroma) -> int:
    """Number of documents in the collection."""
    return store._collection.count()  # noqa: SLF001 - no public count() on Chroma
