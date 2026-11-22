"""Text normalisation and boilerplate removal.

Order matters: NFKC first (fixes ligatures), then header/footer and
table-of-contents removal, then whitespace collapsing.
"""

from __future__ import annotations

import re
import unicodedata

from src.config import (
    DOT_LEADER_MIN_DOTS,
    PAGE_HEADER,
    PAGE_HEADER_SUFFIX,
)
from src.loading.pdf_loader import Line, Page

# Characters NFKC usually handles; kept explicit for safety and documentation.
LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}

# Zone thresholds (fraction of page height) for running header / footer.
HEADER_ZONE = 0.06
FOOTER_ZONE = 0.90

_DOT_LEADER_RE = re.compile(r"\.\s*\.\s*\.")  # at least three spaced dots
_TRAILING_PAGE_RE = re.compile(r"(\d{1,4})\s*$")
_SPACE_RE = re.compile(r"[ \t\u00a0]+")


def normalize_text(text: str) -> str:
    """NFKC-normalize, repair ligature glyphs and collapse spaces."""
    text = unicodedata.normalize("NFKC", text).translate(
        {ord(glyph): replacement for glyph, replacement in LIGATURES.items()}
    )
    # Discretionary hyphens become real hyphens after NFKC.
    return _SPACE_RE.sub(" ", text).strip()


def is_running_header(line: Line, page: Page) -> bool:
    """Top-of-page banner such as 'Overview of Amazon Web Services AWS Whitepaper'."""
    if line.y0 > page.height * HEADER_ZONE:
        return False
    return PAGE_HEADER in line.text or PAGE_HEADER_SUFFIX in line.text


def is_running_footer(line: Line, page: Page) -> bool:
    """Bottom-of-page line: section name followed by the printed page number."""
    return line.y0 > page.height * FOOTER_ZONE


def footer_page_label(line: Line) -> str | None:
    """Return the printed page number from a footer line, if present."""
    match = _TRAILING_PAGE_RE.search(line.text)
    return match.group(1) if match else None


def is_dot_leader_line(text: str) -> bool:
    """Detect table-of-contents lines like 'Amazon EC2 .......... 39'."""
    if not _TRAILING_PAGE_RE.search(text):
        return False
    dot_count = text.count(".")
    if dot_count >= DOT_LEADER_MIN_DOTS:
        return True
    return bool(_DOT_LEADER_RE.search(text))


def is_page_number_line(text: str) -> bool:
    """A bare number left over from a footer."""
    return bool(re.fullmatch(r"\d{1,4}", text.strip()))


class CleanPage:
    """Cleaned lines for one page plus its printed (book) page label."""

    def __init__(self, number: int, lines: list[Line], printed_label: str | None):
        self.number = number
        self.lines = lines
        self.printed_label = printed_label

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CleanPage({self.number}, {len(self.lines)} lines)"


def clean_page(page: Page, *, is_toc_page: bool) -> CleanPage | None:
    """Remove boilerplate from one page. Returns None for TOC pages."""
    if is_toc_page:
        return None

    printed_label: str | None = None
    kept: list[Line] = []

    for line in page.lines:
        text = normalize_text(line.text)
        if not text:
            continue

        if is_running_header(line, page):
            continue

        if is_running_footer(line, page):
            label = footer_page_label(line)
            if label:
                printed_label = label
            continue

        if is_page_number_line(text) or is_dot_leader_line(text):
            continue

        kept.append(
            Line(
                text=text,
                x0=line.x0,
                y0=line.y0,
                x1=line.x1,
                y1=line.y1,
                fonts=line.fonts,
                block=line.block,
            )
        )

    return CleanPage(number=page.number, lines=kept, printed_label=printed_label)


def clean_pages(pages: list[Page], toc_pages: set[int]) -> list[CleanPage]:
    """Clean every page, dropping TOC pages entirely."""
    cleaned: list[CleanPage] = []
    for page in pages:
        result = clean_page(page, is_toc_page=page.number in toc_pages)
        if result is not None:
            cleaned.append(result)
    return cleaned
