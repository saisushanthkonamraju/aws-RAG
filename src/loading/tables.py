"""Turn the 'Compare AWS ... services' tables into plain sentences.

The tables lose their cell boundaries in plain text extraction, so we rebuild
rows from line coordinates, rejoin words that were broken by cell wrapping and
emit one grounded sentence per service row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.loading.cleaner import normalize_text
from src.loading.pdf_loader import Line

TABLE_TITLE_RE = re.compile(r"^Compare AWS .+ services$")

_HEADER_WORDS = {"category", "aws service", "database", "use cases", "aws services"}
_HYPHEN_END_RE = re.compile(r"(?<![-\s])-$")
_TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?)\]}])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{])\s+")

# Suffixes that can be orphaned on a wrapped line (e.g. "unparalle|led").
_SUFFIX_FRAGMENTS = {
    "led",
    "ing",
    "ed",
    "ion",
    "tion",
    "sion",
    "ment",
    "ity",
    "ance",
    "ence",
    "ons",
    "ers",
    "ally",
    "lity",
    "ce",
    "es",
    "s",
    "t",
}


def is_table_title(title: str) -> bool:
    """True for outline entries that introduce a comparison table."""
    return bool(TABLE_TITLE_RE.match(title))


def build_vocabulary(texts: list[str]) -> set[str]:
    """Lowercase word set from regular prose, used to detect split words."""
    vocab: set[str] = set()
    for text in texts:
        vocab.update(_TOKEN_RE.findall(text.lower()))
    return vocab


def _alpha(token: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", token.lower())


def _join_fragments(previous: str, following: str, vocab: set[str]) -> str:
    """Join two wrapped cell lines, guessing whether the break split a word."""
    if _HYPHEN_END_RE.search(previous):
        # "cost-" + "effective" -> "cost-effective"
        return previous + normalize_text(following)

    last = _alpha(previous.rsplit(" ", 1)[-1])
    first = _alpha(following.split(" ", 1)[0])

    if last and first:
        joined = last + first
        hyphen_piece = last.split("-")[-1]

        if joined in vocab and (last not in vocab or first not in vocab):
            return previous + normalize_text(following)

        if (
            hyphen_piece != last
            and hyphen_piece + first in vocab
            and first not in vocab
        ):
            return previous + normalize_text(following)

        # Unknown compound: a long fragment followed by a suffix is a split.
        if last not in vocab and len(last) >= 5 and first in _SUFFIX_FRAGMENTS:
            return previous + normalize_text(following)

    return previous + " " + normalize_text(following)


def _join_cell_lines(lines: list[str], vocab: set[str]) -> str:
    text = normalize_text(lines[0])
    for line in lines[1:]:
        text = _join_fragments(text, line, vocab)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class TableRow:
    """One table row: a category label plus one or more service entries."""

    label: str
    middle: list[str] = field(default_factory=list)
    service_lines: list[list[str]] = field(default_factory=list)


def _header_columns(lines: list[Line]) -> tuple[int, list[float]]:
    """Find the header row and return (line index after header, column x's)."""
    for index, line in enumerate(lines):
        if not line.is_bold or line.text.lower() not in _HEADER_WORDS:
            continue
        y = line.y0
        columns = sorted(
            other.x0
            for other in lines
            if abs(other.y0 - y) < 2.5 and other.text.lower() in _HEADER_WORDS
        )
        if len(columns) >= 2:
            last_header_index = max(
                i for i, other in enumerate(lines) if abs(other.y0 - y) < 2.5
            )
            return last_header_index + 1, columns
    return 0, []


def _parse_rows(lines: list[Line]) -> list[TableRow]:
    """Group table lines into rows using their x coordinates."""
    start, columns = _header_columns(lines)
    if not columns:
        return []

    rows: list[TableRow] = []
    current: TableRow | None = None
    last_column = len(columns) - 1

    for line in lines[start:]:
        if line.text.lower() in _HEADER_WORDS:
            continue
        column = max(
            (i for i, x in enumerate(columns) if line.x0 >= x - 6),
            default=0,
        )
        if column == 0:
            current = TableRow(label=line.text)
            rows.append(current)
            continue
        if current is None:
            current = TableRow(label="")
            rows.append(current)
        if column == last_column:
            if line.text.startswith("•"):
                current.service_lines.append([line.text.lstrip("• ").strip()])
            elif current.service_lines:
                current.service_lines[-1].append(line.text)
            else:
                current.service_lines.append([line.text])
        else:
            current.middle.append(line.text)
    return rows


def _short_table_name(title: str) -> str:
    """'Compare AWS compute services' -> 'Compute'."""
    name = title.removeprefix("Compare AWS ").removesuffix(" services")
    return name[:1].upper() + name[1:]


def table_to_sentences(title: str, lines: list[Line], vocab: set[str]) -> str:
    """Rewrite every service entry of a comparison table as one sentence."""
    sentences: list[str] = []
    table_name = _short_table_name(title)

    for row in _parse_rows(lines):
        label = _join_cell_lines([row.label], vocab) if row.label else table_name
        middle = _join_cell_lines(row.middle, vocab) if row.middle else ""

        prefix = f"{table_name} / {label}" if label != table_name else table_name
        if middle:
            prefix = f"{prefix} (use cases: {middle})"

        for service_lines in row.service_lines:
            text = _join_cell_lines(service_lines, vocab)
            if not text:
                continue
            if "—" in text:
                name, description = (part.strip() for part in text.split("—", 1))
                sentence = f"{prefix}: {name} — {description}"
            else:
                sentence = f"{prefix}: {text}"
            if not sentence.endswith((".", "!", "?")):
                sentence += "."
            sentences.append(sentence)

    return "\n".join(sentences)
