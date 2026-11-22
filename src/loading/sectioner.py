"""Split the whitepaper into service sections using the PDF outline.

The outline (299 bookmarks) is the source of truth for the hierarchy; bold
heading lines in the text tell us exactly where each section starts, including
headings that begin mid-page.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import src.config as config
from src.loading.cleaner import CleanPage, normalize_text
from src.loading.pdf_loader import Line, OutlineEntry
from src.loading.tables import build_vocabulary, is_table_title, table_to_sentences

GENERIC_ROOT = "AWS services by category"
MAX_PAGE_DRIFT = 2  # headings may start a page after their bookmark target
FUZZY_THRESHOLD = 0.88


@dataclass
class Section:
    """One grounded unit of the whitepaper."""

    id: str
    title: str
    category: str
    service: str
    section_path: str
    page_start: int
    page_end: int
    printed_page_start: str | None
    printed_page_end: str | None
    doc_type: str
    text: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "service": self.service,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "printed_page_start": self.printed_page_start,
            "printed_page_end": self.printed_page_end,
            "doc_type": self.doc_type,
            "text": self.text,
        }


@dataclass(order=True)
class HeadingCandidate:
    """A bold line (or wrapped group of bold lines) in the document."""

    index: int
    page: int
    text: str = field(compare=False)


@dataclass
class SectionReport:
    """Diagnostics for the ingest stage."""

    unmatched_outline: list[OutlineEntry] = field(default_factory=list)
    fuzzy_matches: list[tuple[str, str]] = field(default_factory=list)
    empty_sections: list[str] = field(default_factory=list)


@dataclass
class RawSection:
    """Section text before the table-repair pass."""

    entry: OutlineEntry
    doc_type: str
    body: list[tuple[int, Line]]
    ancestors: list[tuple[int, str]]
    page_end: int


def _norm_title(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _heading_candidates(
    flat: list[tuple[int, Line]], min_page: int
) -> list[HeadingCandidate]:
    """Collect bold headings, merging headings that wrap onto a second line."""
    candidates: list[HeadingCandidate] = []
    for index, (page, line) in enumerate(flat):
        if page < min_page or not line.text or not line.is_bold:
            continue
        if candidates and candidates[-1].page == page:
            previous = candidates[-1]
            previous_line = flat[previous.index][1]
            gap = line.y0 - previous_line.y0
            if 0 < gap < 26 and not previous.text.endswith("."):
                previous.text = f"{previous.text} {line.text}"
                continue
        candidates.append(HeadingCandidate(index=index, page=page, text=line.text))
    return candidates


def _align(
    entries: list[OutlineEntry],
    candidates: list[HeadingCandidate],
    report: SectionReport,
) -> list[tuple[OutlineEntry, HeadingCandidate]]:
    """Match outline entries to heading candidates in document order."""
    matched: list[tuple[OutlineEntry, HeadingCandidate]] = []
    cursor = 0

    for entry in entries:
        target = _norm_title(entry.title)
        exact: int | None = None
        fuzzy: int | None = None
        fuzzy_text = ""

        index = cursor
        while index < len(candidates) and candidates[index].page <= entry.page + MAX_PAGE_DRIFT:
            candidate = candidates[index]
            candidate_title = _norm_title(candidate.text)
            if candidate_title == target:
                exact = index
                break
            if fuzzy is None:
                score = difflib.SequenceMatcher(None, target, candidate_title).ratio()
                if score >= FUZZY_THRESHOLD:
                    fuzzy = index
                    fuzzy_text = candidate.text
            index += 1

        chosen = exact if exact is not None else fuzzy
        if chosen is None:
            report.unmatched_outline.append(entry)
            continue

        if exact is None:
            report.fuzzy_matches.append((entry.title, fuzzy_text))

        matched.append((entry, candidates[chosen]))
        cursor = chosen + 1

    return matched


def _starts_new_paragraph(previous_page: int, previous: Line, page: int, line: Line) -> bool:
    """Paragraph breaks: headings, bullets, larger gaps, or a new page after a full stop."""
    if line.is_bold:
        return True
    if previous.x0 - line.x0 > 10:
        return True
    if line.text.startswith("•"):
        return True
    if page == previous_page:
        return line.y0 - previous.y0 > 26
    return previous.text.rstrip().endswith((".", "!", "?", ":")) and not line.text[:1].islower()


def _join_paragraphs(body: list[tuple[int, Line]]) -> str:
    """Group lines into paragraphs using vertical spacing and punctuation."""
    paragraphs: list[str] = []
    current = ""
    previous_page = 0
    previous: Line | None = None

    for page, line in body:
        if current and previous is not None and _starts_new_paragraph(previous_page, previous, page, line):
            paragraphs.append(current)
            current = ""
        if not current:
            current = line.text
        elif current.endswith(("-", "—")):
            current += line.text
        else:
            current = f"{current} {line.text}"
        previous_page, previous = page, line

    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs)


def _classify(depth: int, ancestors: list[tuple[int, str]], title: str) -> str:
    """Map an outline entry to service / category_intro / comparison_table / concept."""
    if is_table_title(title):
        return "comparison_table"
    if depth == 0:
        return "concept"
    if depth == 1 and ancestors and ancestors[-1][1] == GENERIC_ROOT:
        return "category_intro"
    if any(name == GENERIC_ROOT for _, name in ancestors):
        return "service" if depth >= 2 else "category_intro"
    return "concept"


def _section_path(stack: list[tuple[int, str]], title: str) -> str:
    ancestors = [name for depth, name in stack if depth > 0 and name != GENERIC_ROOT]
    return " > ".join([*ancestors, title])


def build_sections(
    pages: list[CleanPage],
    outline: list[OutlineEntry],
    *,
    content_start: int,
) -> tuple[list[Section], SectionReport]:
    """Build one Section per outline entry, in outline order."""
    report = SectionReport()

    flat: list[tuple[int, Line]] = [(page.number, line) for page in pages for line in page.lines]
    candidates = _heading_candidates(flat, min_page=content_start)
    entries = [entry for entry in outline if entry.page >= content_start]
    matched = _align(entries, candidates, report)

    labels = {page.number: page.printed_label for page in pages}
    raw: list[RawSection] = []
    stack: list[tuple[int, str]] = []

    for position, (entry, heading) in enumerate(matched):
        start = heading.index
        end = matched[position + 1][1].index if position + 1 < len(matched) else len(flat)
        body = flat[start + 1 : end]

        while stack and stack[-1][0] >= entry.depth:
            stack.pop()
        ancestors = list(stack)
        stack.append((entry.depth, entry.title))

        raw.append(
            RawSection(
                entry=entry,
                doc_type=_classify(entry.depth, ancestors, entry.title),
                body=body,
                ancestors=ancestors,
                page_end=body[-1][0] if body else entry.page,
            )
        )

    # Prose first: table repairs need a vocabulary of correctly spelled words.
    prose_vocab = build_vocabulary(
        [
            _join_paragraphs(section.body)
            for section in raw
            if section.doc_type != "comparison_table"
        ]
    )

    sections: list[Section] = []
    for counter, section in enumerate(raw):
        if section.doc_type == "comparison_table":
            table_lines = [line for _, line in section.body]
            text = table_to_sentences(section.entry.title, table_lines, prose_vocab)
        else:
            text = _join_paragraphs(section.body)

        if not text.strip():
            report.empty_sections.append(section.entry.title)

        category = next(
            (name for depth, name in section.ancestors if depth == 1 and name != GENERIC_ROOT),
            section.entry.title if section.doc_type == "category_intro" else "",
        )
        sections.append(
            Section(
                id=f"s{counter:03d}",
                title=section.entry.title,
                category=category,
                service=section.entry.title if section.doc_type == "service" else "",
                section_path=_section_path(section.ancestors, section.entry.title),
                page_start=section.entry.page,
                page_end=section.page_end,
                printed_page_start=labels.get(section.entry.page),
                printed_page_end=labels.get(section.page_end),
                doc_type=section.doc_type,
                text=text,
            )
        )

    return sections, report
