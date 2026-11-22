"""Scratch QA: scan processed sections for leftovers from bad extraction."""
import json
import re

with open("data/processed/sections.json", encoding="utf-8") as handle:
    sections = json.load(handle)

problems = {
    "ligatures": re.compile(r"[\ufb00-\ufb06]"),
    "running_header": re.compile(r"AWS Whitepaper"),
    "dot_leaders": re.compile(r"\.{5,}"),
    "double_period": re.compile(r"\.\."),
    "broken_dash": re.compile(r"—\s*$", re.M),
}

for name, pattern in problems.items():
    hits = []
    for section in sections:
        for match in pattern.finditer(section["text"]):
            start = max(0, match.start() - 45)
            hits.append((section["section_path"], section["text"][start : match.end() + 45]))
            break
    print(f"{name}: {len(hits)}")
    for path, sample in hits[:8]:
        print(f"   {path}: ...{sample.strip()}...")

print("\nsections:", len(sections), "with text:", sum(1 for s in sections if s["text"].strip()))
print("total chars:", sum(len(s["text"]) for s in sections))
print("unique ids:", len({s["id"] for s in sections}))
