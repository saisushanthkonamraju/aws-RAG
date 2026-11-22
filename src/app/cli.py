"""Interactive ask-a-question CLI."""

from __future__ import annotations

import src.config as config
from src.generation.chain import Answer, answer_question


def print_answer(answer: Answer) -> None:
    """Print the answer followed by its retrieved sources."""
    print("\nAnswer:\n" + answer.text)

    if not answer.sources:
        return
    print("\nSources:")
    for number, chunk in enumerate(answer.sources, start=1):
        metadata = chunk.document.metadata
        start = metadata.get("printed_page_start") or metadata.get("page_start")
        end = metadata.get("printed_page_end") or metadata.get("page_end")
        pages = f"p. {start}" if start == end else f"pp. {start}-{end}"
        detail = (
            "parent section (added context)"
            if chunk.origin == "parent"
            else f"score {chunk.score:.3f}"
        )
        print(
            f"  [{number}] {metadata['section_path']}  |  {pages}"
            f"  |  {detail}"
        )


def run_cli() -> None:
    """Read questions until the user exits."""
    print(f"AWS whitepaper RAG ({config.LLM_MODEL}). Type 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            print_answer(answer_question(question))
        except Exception as error:  # keep the session alive on API/quota errors
            print(f"[error] {error}")
