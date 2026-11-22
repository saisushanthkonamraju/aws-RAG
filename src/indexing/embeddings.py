"""Local BGE embeddings.

Documents and queries share the same model; the BGE query instruction is added
by the retriever, never here.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from langchain_huggingface import HuggingFaceEmbeddings

import src.config as config


def resolve_device() -> str:
    """Pick CUDA when available, otherwise CPU."""
    if config.EMBEDDING_DEVICE != "auto":
        return config.EMBEDDING_DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the configured embedding model once per process."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": resolve_device()},
        encode_kwargs={"normalize_embeddings": config.NORMALIZE_EMBEDDINGS},
    )
