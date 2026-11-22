"""Stage 1: load the PDF, clean it and split it into sections.

Output: data/processed/sections.json plus a human-readable summary.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.loading.cleaner import clean_pages
from src.loading.pdf_loader import OutlineEntry, load_pages, read_outline
from src.loading.sectioner import Section, build_sections


def find_toc_pages(outline: list[OutlineEntry]) -> set[int]:
    """Pages occupied by the table of contents: its bookmark until the next one."""
    for index, entry in enumerate(outline):
        if entry.title == config.OUTLINE_TOC_TITLE:
            following = next(
                (other.page for other in outline[index + 1 :] if other.page > entry.page),
                entry.page + 1,
            )
            return set(range(entry.page, following))
    return set()


def print_ranked(title: str, sections: list[Section]) -> None:
    ordered = sorted(sections, key=lambda section: len(section.text))
    print(f"\n{title} (characters, page range):")
    for section in ordered[:10] if "shortest" in title else ordered[-10:]:
        print(
            f"  {len(section.text):6d}  p{section.page_start}-{section.page_end}"
            f"  [{section.doc_type}]  {section.section_path}"
        )


def main() -> None:
    print(f"PDF: {config.PDF_PATH}")
    pages = load_pages(config.PDF_PATH)
    outline = read_outline(config.PDF_PATH)
    print(f"Loaded {len(pages)} pages and {len(outline)} outline entries")

    toc_pages = find_toc_pages(outline)
    content_start = max(toc_pages) + 1 if toc_pages else 1
    print(f"Table of contents pages removed: {sorted(toc_pages)}")

    cleaned = clean_pages(pages, toc_pages)
    print(f"Cleaned {len(cleaned)} pages")

    sections, report = build_sections(cleaned, outline, content_start=content_start)

    counts = Counter(section.doc_type for section in sections)
    print("\nSections by type:")
    for doc_type, count in counts.most_common():
        print(f"  {doc_type:18} {count}")

    print(f"\nSections with no text: {len(report.empty_sections)}")
    for title in report.empty_sections:
        print(f"  {title}")

    print(f"\nOutline titles that could not be located: {len(report.unmatched_outline)}")
    for entry in report.unmatched_outline:
        print(f"  depth {entry.depth}  p{entry.page}  {entry.title}")

    print(f"\nFuzzy heading matches: {len(report.fuzzy_matches)}")
    for wanted, found in report.fuzzy_matches:
        print(f"  {wanted!r} -> {found!r}")

    print_ranked("10 shortest sections", sections)
    print_ranked("10 longest sections", sections)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    payload = [section.to_dict() for section in sections]
    config.SECTIONS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {len(payload)} sections to {config.SECTIONS_PATH}")


if __name__ == "__main__":
    main()
