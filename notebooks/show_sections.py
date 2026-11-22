"""Scratch helper: print saved sections by title (substring match)."""
import json
import sys

with open("data/processed/sections.json", encoding="utf-8") as handle:
    sections = json.load(handle)

needles = [arg.lower() for arg in sys.argv[1:]]
for section in sections:
    if any(needle in section["title"].lower() or needle in section["section_path"].lower() for needle in needles):
        print("=" * 78)
        print(f"{section['section_path']}  |  {section['doc_type']}")
        print(
            f"pages {section['page_start']}-{section['page_end']}"
            f" (printed {section['printed_page_start']}-{section['printed_page_end']})"
            f"  chars={len(section['text'])}"
        )
        print("-" * 78)
        print(section["text"])
        print()
