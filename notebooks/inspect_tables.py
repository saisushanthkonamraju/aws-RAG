"""Scratch helper: test PyMuPDF table extraction on the comparison pages."""
import json
import sys

import pymupdf

doc = pymupdf.open(sys.argv[1])
for pno in [int(x) for x in sys.argv[2:]]:
    page = doc[pno - 1]
    print(f"================ PAGE {pno} ================")
    tabs = page.find_tables()
    print("tables:", len(tabs.tables))
    for t in tabs.tables:
        for row in t.extract():
            print(json.dumps(row, ensure_ascii=False))
