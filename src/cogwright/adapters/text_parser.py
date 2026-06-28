# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""A parser for plain-text and Markdown manuals.

It recovers the structure the chunker relies on (headings, procedure steps, and
pipe tables) using simple, well-documented heuristics, and treats form-feed
characters as page breaks so citations carry real page numbers. The heuristics
are intentionally conservative; richer layout recovery for born-digital PDFs
lives in the PDF parser, and scanned-page understanding is future work.
"""

from __future__ import annotations

import os
import re

from ..core.models import BlockKind, Document, TextBlock

_STEP = re.compile(r"^\s*(?:\d+[.)]\s+|step\s+\d+\b)", re.IGNORECASE)
_ATX_HEADING = re.compile(r"^\s*#{1,6}\s+(?P<text>.+?)\s*#*\s*$")
_SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")

# A run of spaced dots or dashes is a table-of-contents leader.
_DOT_LEADER = re.compile(r"(?:[.·]\s?){4,}|(?:-\s){3,}")
# A trailing "chapter-page" reference, e.g. "4-111", "6-205", or "EDITOR4-133"
# where the title and number ran together, marks a contents entry or a running
# page header rather than a real heading.
_TRAILING_PAGE_REF = re.compile(r"\d+\s*[-–]\s*\d+\s*$")


class TextDocumentParser:
    """Parses ``.txt``, ``.text``, and ``.md`` documents into normalized blocks."""

    _EXTENSIONS = (".txt", ".text", ".md", ".markdown")

    def supports(self, path: str) -> bool:
        return path.lower().endswith(self._EXTENSIONS)

    def parse(self, path: str, data: bytes) -> Document:
        text = data.decode("utf-8", errors="replace")
        stem = os.path.splitext(os.path.basename(path))[0]
        blocks: list[TextBlock] = []
        title: str | None = None

        # A form feed starts a new page; everything else is page 1 by default.
        for page_number, page_text in enumerate(text.split("\f"), start=1):
            page_blocks = parse_lines(page_text, page_number)
            for block in page_blocks:
                if title is None and block.kind == BlockKind.HEADING:
                    title = block.text
            blocks.extend(page_blocks)

        return Document(
            document_id=stem,
            source_path=path,
            title=title or stem,
            blocks=tuple(blocks),
        )


def parse_lines(page_text: str, page: int) -> list[TextBlock]:
    """Parse one page of text into structured blocks.

    Shared with the PDF parser, which feeds it the text it extracts per page.
    """

    lines = page_text.splitlines()
    blocks: list[TextBlock] = []
    paragraph: list[str] = []
    i = 0

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(
                TextBlock(
                    kind=BlockKind.PARAGRAPH,
                    text=" ".join(paragraph).strip(),
                    page=page,
                )
            )
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        # Drop table-of-contents lines outright; they are navigation, not content.
        if _is_navigation_line(stripped):
            flush_paragraph()
            i += 1
            continue

        if "|" in line and _looks_like_table_row(line):
            flush_paragraph()
            table_lines: list[str] = []
            while i < len(lines) and "|" in lines[i] and _looks_like_table_row(lines[i]):
                table_lines.append(lines[i])
                i += 1
            block = _build_table(table_lines, page)
            if block is not None:
                blocks.append(block)
            continue

        heading = _heading_text(stripped)
        if heading is not None:
            flush_paragraph()
            blocks.append(TextBlock(kind=BlockKind.HEADING, text=heading, page=page))
            i += 1
            continue

        if _STEP.match(line):
            flush_paragraph()
            blocks.append(TextBlock(kind=BlockKind.STEP, text=stripped, page=page))
            i += 1
            continue

        paragraph.append(stripped)
        i += 1

    flush_paragraph()
    return blocks


def _heading_text(stripped: str) -> str | None:
    atx = _ATX_HEADING.match(stripped)
    if atx:
        return atx.group("text").strip()
    # Table-of-contents entries and running page headers ("EDITOR 4-111",
    # "INTRODUCTION - - - - 1-1") are all-caps too, so they would otherwise be
    # mistaken for headings and pollute the section context. Reject them.
    if _DOT_LEADER.search(stripped) or _TRAILING_PAGE_REF.search(stripped):
        return None
    # An all-caps line with no terminal punctuation reads as a section heading,
    # which matches how plain-text equipment manuals are typically laid out.
    has_letter = any(c.isalpha() for c in stripped)
    if (
        has_letter
        and 2 <= len(stripped) <= 60
        and stripped == stripped.upper()
        and not stripped.endswith((".", ":", ",", ";"))
    ):
        return stripped
    return None


def _is_navigation_line(stripped: str) -> bool:
    """Whether a line is contents-page navigation rather than content."""

    return bool(_DOT_LEADER.search(stripped) and _TRAILING_PAGE_REF.search(stripped))


def _looks_like_table_row(line: str) -> bool:
    # Require at least two cells so a single inline pipe is not mistaken for a row.
    return "|" in line and len(_split_row(line)) >= 2


def _split_row(line: str) -> list[str]:
    cells = [cell.strip() for cell in line.split("|")]
    # Drop the empty cells produced by leading and trailing pipes.
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _build_table(table_lines: list[str], page: int) -> TextBlock | None:
    rows: list[tuple[str, ...]] = []
    for line in table_lines:
        cells = _split_row(line)
        if all(_SEPARATOR_CELL.match(cell) for cell in cells if cell):
            # Markdown header separator row, e.g. "--- | ---"; skip it.
            continue
        rows.append(tuple(cells))
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return None
    text = "\n".join(" | ".join(row) for row in rows)
    return TextBlock(
        kind=BlockKind.TABLE,
        text=text,
        page=page,
        rows=tuple(rows),
    )
