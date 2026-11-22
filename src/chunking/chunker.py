"""Section-aware chunking.

One chunk per section by default. Only sections longer than the configured
limit are split, so chunks never cross a service boundary. Every chunk text is
prefixed with its short hierarchical header.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

import src.config as config


@dataclass
class Chunk:
    """A chunk ready for embedding, with the required metadata."""

    id: str
    text: str
    metadata: dict

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "metadata": self.metadata}


def chunk_header(section_path: str) -> str:
    """Short context line embedded with every chunk."""
    return f"{config.SECTION_HEADER_PREFIX} > {section_path}"


def build_chunks(sections: list[dict]) -> list[Chunk]:
    """Turn processed sections into embeddable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks: list[Chunk] = []
    for section in sections:
        text = section["text"].strip()
        if not text:
            continue

        pieces = (
            [text]
            if len(text) <= config.CHUNK_SIZE
            else splitter.split_text(text)
        )
        pieces = [piece.strip() for piece in pieces if piece.strip()]

        header = chunk_header(section["section_path"])
        for index, piece in enumerate(pieces):
            if len(piece) < config.MIN_CHUNK_CHARS and len(pieces) > 1:
                continue
            chunks.append(
                Chunk(
                    id=f"{section['id']}_c{index:02d}",
                    text=f"{header}\n\n{piece}",
                    metadata={
                        "section_id": section["id"],
                        "section_path": section["section_path"],
                        "title": section["title"],
                        "category": section["category"],
                        "service": section["service"],
                        "doc_type": section["doc_type"],
                        "page_start": int(section["page_start"]),
                        "page_end": int(section["page_end"]),
                        "printed_page_start": section["printed_page_start"] or "",
                        "printed_page_end": section["printed_page_end"] or "",
                        "chunk_index": index,
                    },
                )
            )

    return chunks
