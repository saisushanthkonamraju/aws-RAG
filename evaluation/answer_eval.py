"""Answer-level evaluation: citation validity and refusal behaviour.

Calls the real RAG chain on every golden question, then checks each citation
in the answer against the chunks that were actually retrieved:

* the cited page must overlap a retrieved chunk's printed page range
* the cited label must match that chunk's section path, title, service or text

It also verifies that unanswerable questions are refused and that answerable
ones are not. Requires GOOGLE_API_KEY; makes one LLM call per question.

    python evaluation/answer_eval.py
    python evaluation/answer_eval.py --ids q01 q05 q14
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.generation.chain import answer_question
from src.retrieval.types import RetrievedChunk

GOLDEN_PATH = Path(__file__).with_name("golden_questions.json")

# "(AWS Lambda, p. 39)" and "(Analytics, pp. 9-11)" inside a parenthetical.
CITATION_RE = re.compile(r"\(([^()]*?(?:p\.|pp\.)\s*\d+[^()]*)\)", re.IGNORECASE)
SEGMENT_RE = re.compile(
    r"^(?P<label>.*?),\s*(?:p\.|pp\.)\s*(?P<start>\d+)(?:\s*[-\u2013]\s*(?P<end>\d+))?\s*$",
    re.IGNORECASE,
)

REFUSAL_MARKERS = (
    "does not contain",
    "does not cover",
    "does not provide",
    "does not specify",
    "does not mention",
    "not covered",
    "not mentioned",
    "not specified",
    "no information",
    "cannot find",
    "can't find",
    "not available in the",
    "doesn't contain",
    "doesn't specify",
)


def load_golden() -> list[dict]:
    """Read the golden question file."""
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["questions"]


def normalize(text: str) -> str:
    """Lowercase and strip punctuation so labels compare to section paths."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def token_set(text: str) -> set[str]:
    """Words of a normalized string, ignoring single-letter fragments."""
    return {token for token in normalize(text).split() if len(token) > 1}


def parse_citations(answer: str) -> list[dict]:
    """Split an answer's parentheticals into individual label/page citations."""
    citations: list[dict] = []
    for match in CITATION_RE.finditer(answer):
        for segment in match.group(1).split(";"):
            parsed = SEGMENT_RE.match(segment.strip())
            if not parsed:
                continue
            start = int(parsed.group("start"))
            end = int(parsed.group("end") or start)
            citations.append(
                {"label": parsed.group("label").strip(), "start": start, "end": end}
            )
    return citations


def page_span(chunk: RetrievedChunk) -> tuple[int, int] | None:
    """Printed page range of a retrieved chunk, if it has one."""
    metadata = chunk.document.metadata
    start = metadata.get("printed_page_start") or metadata.get("page_start")
    end = metadata.get("printed_page_end") or metadata.get("page_end") or start
    if start is None:
        return None
    try:
        return int(start), int(end if end is not None else start)
    except (TypeError, ValueError):
        return None


def validate(citation: dict, chunks: list[RetrievedChunk]) -> tuple[bool, str]:
    """A citation is valid when a retrieved chunk covers both label and pages."""
    label_tokens = token_set(citation["label"])
    if not label_tokens:
        return False, "empty label"

    for chunk in chunks:
        span = page_span(chunk)
        if not span:
            continue
        start, end = span
        if citation["end"] < start or citation["start"] > end:
            continue

        metadata = chunk.document.metadata
        haystack = " ".join(
            str(part)
            for part in (
                metadata.get("section_path", ""),
                metadata.get("title", ""),
                metadata.get("service", ""),
                chunk.document.page_content[:400],
            )
        )
        if label_tokens <= token_set(haystack):
            return True, "matched"
    return False, "no retrieved chunk has this label at these pages"


def requested_refusal(answer: str) -> bool:
    """Heuristic: does the answer say the document does not cover the topic?"""
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def evaluate_one(item: dict, k: int) -> dict:
    """Run the chain on one question and check its citations."""
    answer = answer_question(item["question"], k=k)
    citations = parse_citations(answer.text)
    checks = [(citation, *validate(citation, answer.sources)) for citation in citations]
    invalid = [citation for citation, ok, _ in checks if not ok]
    return {
        "id": item["id"],
        "question": item["question"],
        "answerable": item["answerable"],
        "answer": answer.text,
        "citations": citations,
        "invalid": invalid,
        "refused": requested_refusal(answer.text),
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate citation precision and refusal accuracy."""
    all_citations = [citation for row in rows for citation in row["citations"]]
    invalid = [citation for row in rows for citation in row["invalid"]]

    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    cited_answerable = [row for row in answerable if row["citations"]]

    return {
        "citations": len(all_citations),
        "invalid": len(invalid),
        "precision": (
            (len(all_citations) - len(invalid)) / len(all_citations)
            if all_citations
            else 0.0
        ),
        "citation_rate": len(cited_answerable) / len(answerable) if answerable else 0.0,
        "clean_answers": (
            sum(1 for row in answerable if row["citations"] and not row["invalid"])
            / len(answerable)
            if answerable
            else 0.0
        ),
        "refusal_accuracy": (
            sum(1 for row in unanswerable if row["refused"]) / len(unanswerable)
            if unanswerable
            else 0.0
        ),
        "unanswerable": len(unanswerable),
        "false_refusals": [row["id"] for row in answerable if row["refused"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", help="Only evaluate these question ids.")
    parser.add_argument("--show-answers", action="store_true", help="Print each answer.")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds between questions, to stay inside free-tier rate limits.",
    )
    parser.add_argument("--k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    items = load_golden()
    if args.ids:
        wanted = set(args.ids)
        items = [item for item in items if item["id"] in wanted]
        if not items:
            raise SystemExit(f"No golden questions matched {sorted(wanted)}")

    print(f"Answer-level evaluation: {len(items)} questions, k={args.k}\n")
    rows: list[dict] = []
    for index, item in enumerate(items):
        if index:
            time.sleep(args.delay)
        # Print as we go: an API or network failure never loses earlier results.
        try:
            row = evaluate_one(item, args.k)
        except Exception as error:  # noqa: BLE001 - report and keep going
            print(f"{item['id']}  ERROR     {error}", flush=True)
            continue
        rows.append(row)
        status = "refused" if row["refused"] else "answered"
        valid = len(row["citations"]) - len(row["invalid"])
        print(
            f"{row['id']}  {status:<8}  citations {valid}/{len(row['citations'])} valid",
            flush=True,
        )
        for citation in row["invalid"]:
            print(
                f"     INVALID  ({citation['label']}, pp. {citation['start']}-{citation['end']})",
                flush=True,
            )
        if args.show_answers:
            print("     " + row["answer"].replace("\n", "\n     "), flush=True)

    metrics = summarize(rows)
    print("\nSummary")
    print(f"  citations checked:          {metrics['citations']}")
    print(f"  citation precision:         {metrics['precision']:.3f}")
    print(f"  answerable with citations:  {metrics['citation_rate']:.3f}")
    print(f"  answers fully supported:    {metrics['clean_answers']:.3f}")
    if metrics["unanswerable"]:
        print(f"  refusal accuracy:           {metrics['refusal_accuracy']:.3f}")
    else:
        print("  refusal accuracy:           n/a (no unanswerable questions in this run)")
    if metrics["false_refusals"]:
        print(f"  false refusals:           {', '.join(metrics['false_refusals'])}")


if __name__ == "__main__":
    main()
