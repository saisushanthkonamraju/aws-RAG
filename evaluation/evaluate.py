"""Offline retrieval evaluation: Hit@k, Recall@k, MRR and nDCG@k.

Runs the same golden questions through every retrieval mode, so the effect of
BM25 and the reranker is measurable. No LLM calls, no API key needed.

    python evaluation/evaluate.py
    python evaluation/evaluate.py --k 10
    python evaluation/evaluate.py --modes dense hybrid-rerank
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.retrieval.retriever import search

GOLDEN_PATH = Path(__file__).with_name("golden_questions.json")

MODES = {
    "dense": (False, False, False),
    "mmr": (False, True, False),
    "hybrid": (True, False, False),
    "dense-rerank": (False, False, True),
    "mmr-rerank": (False, True, True),
    "hybrid-rerank": (True, False, True),
}


def load_golden() -> list[dict]:
    """Read the golden question file."""
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["questions"]


def is_hit(section_path: str | None, expected: set[str], mode: str) -> bool:
    """Exact section match, or any descendant when the question is subtree-wide."""
    if not section_path:
        return False
    if section_path in expected:
        return True
    if mode == "subtree":
        return any(section_path.startswith(f"{path} > ") for path in expected)
    return False


def first_hit_rank(results: list, expected: set[str], mode: str) -> int | None:
    """1-based rank of the first retrieved chunk from an expected section."""
    for rank, chunk in enumerate(results, start=1):
        if is_hit(chunk.document.metadata.get("section_path"), expected, mode):
            return rank
    return None


def recall_at_k(results: list, expected: set[str], mode: str) -> float:
    """Fraction of expected sections that appear anywhere in the results."""
    if not expected:
        return 0.0
    retrieved = [chunk.document.metadata.get("section_path") or "" for chunk in results]
    found = sum(
        1 for path in expected if any(is_hit(item, {path}, mode) for item in retrieved)
    )
    return found / len(expected)


def ndcg_at_k(results: list, expected: set[str], k: int, mode: str) -> float:
    """Binary-gain nDCG, counting each expected section at most once."""
    seen: set[str] = set()
    gains: list[float] = []
    for chunk in results[:k]:
        path = chunk.document.metadata.get("section_path") or ""
        matched = next(
            (root for root in expected if is_hit(path, {root}, mode)), None
        )
        if matched is not None and matched not in seen:
            seen.add(matched)
            gains.append(1.0)
        else:
            gains.append(0.0)
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(len(expected), k)))
    return dcg / ideal if ideal else 0.0


def evaluate_mode(
    questions: list[dict],
    use_bm25: bool,
    use_mmr: bool,
    use_reranker: bool,
    k: int,
) -> dict:
    """Run one retrieval configuration over all answerable questions."""
    config.USE_BM25 = use_bm25
    config.USE_MMR = use_mmr
    config.USE_RERANKER = use_reranker

    rows: list[dict] = []
    started = time.perf_counter()
    for item in questions:
        if not item["answerable"]:
            continue
        results = search(item["question"], k=k)
        expected = set(item["expected_sections"])
        mode = item.get("match", "exact")
        rank = first_hit_rank(results, expected, mode)
        rows.append(
            {
                "id": item["id"],
                "hit": rank is not None,
                "recall": recall_at_k(results, expected, mode),
                "mrr": 1.0 / rank if rank else 0.0,
                "ndcg": ndcg_at_k(results, expected, k, mode),
                "rank": rank,
                "sections": len(
                    {
                        chunk.document.metadata.get("section_path")
                        for chunk in results[:k]
                    }
                ),
            }
        )

    elapsed = time.perf_counter() - started
    return {
        "hit": sum(row["hit"] for row in rows) / len(rows),
        "recall": sum(row["recall"] for row in rows) / len(rows),
        "mrr": sum(row["mrr"] for row in rows) / len(rows),
        "ndcg": sum(row["ndcg"] for row in rows) / len(rows),
        "sections": sum(row["sections"] for row in rows) / len(rows),
        "rows": rows,
        "ms_per_query": elapsed * 1000 / len(rows),
    }


def report_unanswerable(questions: list[dict], k: int) -> None:
    """Show what the reranker thinks of questions the document cannot answer."""
    config.USE_BM25 = False
    config.USE_MMR = False
    config.USE_RERANKER = True
    print("\nUnanswerable questions (top-1 rerank score; low = correctly irrelevant)")
    for item in questions:
        if item["answerable"]:
            continue
        top = search(item["question"], k=1)[0]
        metadata = top.document.metadata
        print(
            f"  {item['id']}  {top.score:.3f}  {metadata['section_path']}  |  {item['question']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=config.RERANK_FETCH_K,
        help="Candidates fetched for the reranker before keeping top-k.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge chunks of the same section into one context block.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODES),
        default=list(MODES),
        help="Retrieval modes to compare.",
    )
    args = parser.parse_args()

    questions = load_golden()
    config.RERANK_FETCH_K = args.fetch_k
    config.MERGE_SECTIONS = args.merge
    print(
        f"Golden questions: {len(questions)} "
        f"({sum(q['answerable'] for q in questions)} answerable), "
        f"k={args.k}, rerank fetch={args.fetch_k}, "
        f"merge={'on' if args.merge else 'off'}\n"
    )

    # Warm up embeddings + reranker so per-mode timings reflect steady state.
    config.USE_BM25 = config.USE_MMR = False
    config.USE_RERANKER = True
    search("warm up", k=1)

    header = (
        f"{'mode':<14}{'Hit@k':>8}{'Recall@k':>10}{'MRR':>8}"
        f"{'nDCG@k':>9}{'sec@k':>7}{'ms/query':>10}"
    )
    print(header)
    print("-" * len(header))

    for name in args.modes:
        use_bm25, use_mmr, use_reranker = MODES[name]
        metrics = evaluate_mode(questions, use_bm25, use_mmr, use_reranker, args.k)
        print(
            f"{name:<14}{metrics['hit']:>8.3f}{metrics['recall']:>10.3f}"
            f"{metrics['mrr']:>8.3f}{metrics['ndcg']:>9.3f}"
            f"{metrics['sections']:>7.1f}{metrics['ms_per_query']:>10.0f}"
        )
        misses = [row["id"] for row in metrics["rows"] if not row["hit"]]
        if misses:
            print(f"{'':<14}missed: {', '.join(misses)}")

    report_unanswerable(questions, args.k)


if __name__ == "__main__":
    main()
