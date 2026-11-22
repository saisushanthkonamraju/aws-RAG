"""Grounded answering prompt and context formatting."""

from __future__ import annotations

import re

from src.retrieval.retriever import RetrievedChunk

SYSTEM_PROMPT = """You answer questions about the AWS whitepaper
"Overview of Amazon Web Services" (June 2026 edition).

Style:
- Start directly with the answer. Never open with "According to the whitepaper",
  "Based on the context" or any similar preamble.
- Be concise and factual: state each point once. If several context blocks make
  the same point, merge it and cite the most specific one.

Rules:
1. Answer ONLY from the numbered context blocks provided by the user. Do not use
   outside knowledge, even if you know it.
2. For every claim, cite the service name and page exactly as given in the block
   header, for example "(AWS Lambda, p. 39)". Cite the section or category when
   the block has no specific service.
3. If the context does not contain the answer, say clearly that the whitepaper
   does not cover it. Never guess. This applies especially to pricing, quotas,
   limits and step-by-step instructions.
4. The whitepaper is a June 2026 snapshot; mention that only when recency
   matters.
5. For comparison questions, use only what the context says about each service.
   Do not add differences the document does not state."""

USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

# Openers the model likes to add even when told not to. Matched once at the
# start of the answer (markdown emphasis and whitespace tolerated).
_PREAMBLE_RE = re.compile(
    r"""
    ^[\s*_#>\-]*                        # leading markdown / whitespace
    (?:
        according\s+to\s+(?:the|this|that)\s+
            (?:whitepaper|document|pdf|provided\s+(?:context|text|excerpts?))
      | based\s+on\s+(?:the|this|that|your)\s+
            (?:whitepaper|document|pdf|provided\s+(?:context|text|excerpts?)
               |context|excerpts?)
      | in\s+(?:the|this)\s+(?:whitepaper|document)
      | the\s+(?:whitepaper|document)\s+
            (?:states|says|notes|indicates|explains)(?:\s+that)?
    )
    \s*(?:[*_]{0,2}\s*(?:,|:|;|\.|—|–|-)[*_]{0,2})?\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

_MAX_PREAMBLE_PASSES = 3


def strip_preamble(text: str) -> str:
    """Remove a scripted opener such as 'According to the whitepaper,'.

    The prompt asks for this, but an LLM can still prepend the phrase, so the
    answer is cleaned deterministically before it reaches the user.
    """
    cleaned = text.strip()
    for _ in range(_MAX_PREAMBLE_PASSES):
        stripped = _PREAMBLE_RE.sub("", cleaned, count=1).lstrip()
        if not stripped or stripped == cleaned:
            break
        cleaned = stripped
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _page_label(metadata: dict) -> str:
    start = metadata.get("printed_page_start") or str(metadata.get("page_start"))
    end = metadata.get("printed_page_end") or str(metadata.get("page_end"))
    if start and end and start != end:
        return f"pp. {start}-{end}"
    return f"p. {start}"


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Numbered context blocks with section path and page for each chunk."""
    blocks: list[str] = []
    for number, chunk in enumerate(chunks, start=1):
        metadata = chunk.document.metadata
        path = metadata.get("section_path", metadata.get("title", "Unknown section"))
        body = chunk.document.page_content.split("\n\n", 1)[-1].strip()
        blocks.append(f"[{number}] {path} — {_page_label(metadata)}\n{body}")
    return "\n\n".join(blocks)
