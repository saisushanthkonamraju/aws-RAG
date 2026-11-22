"""Page-by-page PDF text extraction with layout metadata.

PyMuPDF provides text plus font/coordinate information; pypdf provides the
bookmark outline that we treat as the ground-truth section hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pypdf


@dataclass(frozen=True)
class Line:
    """One visual text line, with the metadata needed for heading detection."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    fonts: tuple[str, ...]
    block: int

    @property
    def is_bold(self) -> bool:
        """True when every non-empty span on the line uses a bold font."""
        return bool(self.fonts) and all("Bold" in font for font in self.fonts)


@dataclass
class Page:
    """All lines of one PDF page (1-based page number)."""

    number: int
    width: float
    height: float
    lines: list[Line]


@dataclass(frozen=True)
class OutlineEntry:
    """One PDF bookmark, flattened from the nested outline."""

    depth: int
    title: str
    page: int  # 1-based physical page


def load_pages(pdf_path: str | Path) -> list[Page]:
    """Extract every page as a list of text lines."""
    pages: list[Page] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for page in doc:
            lines: list[Line] = []
            for block_index, block in enumerate(page.get_text("dict")["blocks"]):
                for line in block.get("lines", []):
                    spans = [span for span in line["spans"] if span["text"].strip()]
                    if not spans:
                        continue
                    text = "".join(span["text"] for span in line["spans"])
                    x0, y0, x1, y1 = line["bbox"]
                    lines.append(
                        Line(
                            text=text.strip(),
                            x0=round(x0, 2),
                            y0=round(y0, 2),
                            x1=round(x1, 2),
                            y1=round(y1, 2),
                            fonts=tuple(sorted({span["font"] for span in spans})),
                            block=block_index,
                        )
                    )
            pages.append(
                Page(
                    number=page.number + 1,
                    width=page.rect.width,
                    height=page.rect.height,
                    lines=lines,
                )
            )
    return pages


def read_outline(pdf_path: str | Path) -> list[OutlineEntry]:
    """Flatten the PDF bookmark tree into (depth, title, page) entries."""
    reader = pypdf.PdfReader(str(pdf_path))
    entries: list[OutlineEntry] = []

    def walk(items: list, depth: int = 0) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            try:
                number = reader.get_destination_page_number(item)
            except Exception:  # malformed destination: skip, do not crash
                continue
            if number is None:
                continue
            entries.append(
                OutlineEntry(
                    depth=depth, title=item.title.strip(), page=number + 1
                )
            )

    walk(reader.outline)
    return entries
