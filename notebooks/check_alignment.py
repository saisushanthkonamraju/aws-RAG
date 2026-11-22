"""Scratch helper: compare bold heading lines against the PDF outline."""
import re

import pymupdf
import pypdf

PDF = "data/raw/aws-overview.pdf"

# --- outline ---------------------------------------------------------------
reader = pypdf.PdfReader(PDF)


def walk(items, depth=0):
    for it in items:
        if isinstance(it, list):
            yield from walk(it, depth + 1)
        else:
            yield depth, it


outline = []
for depth, dest in walk(reader.outline):
    outline.append(
        {
            "depth": depth,
            "title": dest.title.strip(),
            "page": reader.get_destination_page_number(dest) + 1,
        }
    )

# --- bold lines ------------------------------------------------------------
doc = pymupdf.open(PDF)
cands = []
for page in doc:
    pno = page.number + 1
    if pno <= 10:
        continue
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text or text == "Overview of Amazon Web Services" or text == "AWS Whitepaper":
                continue
            if re.fullmatch(r"\d+", text):
                continue
            if all("Bold" in s["font"] for s in spans):
                cands.append({"page": pno, "y": round(line["bbox"][1], 1), "text": text})

# merge consecutive bold lines (wrapped headings)
merged = []
for c in cands:
    if (
        merged
        and merged[-1]["page"] == c["page"]
        and 0 < c["y"] - merged[-1]["y"] < 26
        and not merged[-1]["text"].endswith(".")
    ):
        merged[-1]["text"] += " " + c["text"]
        merged[-1]["y"] = c["y"]
    else:
        merged.append(dict(c))


def norm(s: str) -> str:
    s = re.sub(r"[\u2010-\u2015\-]", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


# --- alignment -------------------------------------------------------------
ci = 0
fails = []
for e in outline:
    if e["page"] < 11:
        continue
    hit = None
    for j in range(ci, len(merged)):
        c = merged[j]
        if c["page"] > e["page"] + 1:
            break
        if norm(c["text"]) == norm(e["title"]):
            hit = j
            break
    if hit is None:
        fails.append((e["depth"], e["page"], e["title"]))
    else:
        ci = hit + 1

print("outline entries:", len(outline))
print("bold candidates:", len(merged))
print("exact match failures:", len(fails))
for f in fails:
    print("  MISS", f)
