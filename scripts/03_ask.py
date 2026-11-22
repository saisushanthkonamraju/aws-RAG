"""Quick manual testing: ask one or more questions, or start the CLI.

Examples:
    python scripts/03_ask.py "What is AWS Lambda?"
    python scripts/03_ask.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.app.cli import print_answer, run_cli
from src.generation.chain import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="*", help="Questions to answer; omit for the CLI.")
    parser.add_argument("--k", type=int, default=None, help="Override the number of retrieved chunks.")
    args = parser.parse_args()

    if not args.questions:
        run_cli()
        return

    for index, question in enumerate(args.questions):
        if index:
            print("\n" + "=" * 78)
        print(f"QUESTION: {question}")
        try:
            print_answer(answer_question(question, k=args.k))
        except Exception as error:
            raise SystemExit(f"[error] {error}") from error


if __name__ == "__main__":
    main()
