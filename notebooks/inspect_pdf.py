"""Scratch helper: inspect PDF line/font/table structure."""
import sys

import pymupdf

PDF = sys.argv[1]
pages = [int(x) for x in sys.argv[2:]] or [21]

doc = pymupdf.open(PDF)
for pno in pages:
    page = doc[pno - 1]
    print(f"================ PAGE {pno} ================")
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"])
            fonts = sorted({s["font"] for s in line["spans"]})
            print(f"y{line['bbox'][1]:6.1f} x{line['bbox'][0]:6.1f} {fonts} {text!r}")
