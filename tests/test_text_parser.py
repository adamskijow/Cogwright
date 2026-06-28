# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for the plain-text and Markdown parser heuristics."""

from __future__ import annotations

from pathlib import Path

from cogwright.adapters.text_parser import TextDocumentParser
from cogwright.core.models import BlockKind, Document

FIXTURE = Path(__file__).parent / "fixtures" / "sample_manual" / "series7_conveyor_manual.txt"


def _parse() -> Document:
    data = FIXTURE.read_bytes()
    return TextDocumentParser().parse(str(FIXTURE), data)


def test_headings_steps_and_pages_are_recovered() -> None:
    doc = _parse()
    kinds_by_page: dict[int, set[BlockKind]] = {}
    for block in doc.blocks:
        kinds_by_page.setdefault(block.page, set()).add(block.kind)

    headings = [b.text for b in doc.blocks if b.kind == BlockKind.HEADING]
    assert "STARTUP PROCEDURE" in headings
    assert "ALARM AND STOP CODE REFERENCE" in headings

    # Form feeds produced four pages, and the startup steps are on page 2.
    assert max(b.page for b in doc.blocks) == 4
    assert BlockKind.STEP in kinds_by_page[2]


def test_steps_are_tagged_in_order() -> None:
    doc = _parse()
    page2_steps = [
        b.text for b in doc.blocks if b.page == 2 and b.kind == BlockKind.STEP
    ]
    assert len(page2_steps) == 5
    assert page2_steps[0].startswith("1.")
    assert page2_steps[-1].startswith("5.")


def test_pipe_table_is_parsed_with_rows() -> None:
    doc = _parse()
    tables = [b for b in doc.blocks if b.kind == BlockKind.TABLE]
    assert len(tables) == 1
    table = tables[0]
    assert table.page == 4
    assert table.rows is not None
    # Header plus three component rows; the Markdown separator row is dropped.
    assert len(table.rows) == 4
    assert table.rows[0] == ("Component", "Part Number", "Notes")
    flattened = [cell for row in table.rows for cell in row]
    assert "PN 44-19A" in flattened


def test_contents_and_page_header_lines_are_not_headings() -> None:
    body = (
        "INTRODUCTION - - - - - - - - 1-1\n"
        "EDITOR 4-111\n"
        "EDITOR4-133\n"
        "REAL SECTION HEADING\n"
        "Some body text follows.\n"
    )
    doc = TextDocumentParser().parse("manual.txt", body.encode("utf-8"))
    headings = [b.text for b in doc.blocks if b.kind == BlockKind.HEADING]
    text = "\n".join(b.text for b in doc.blocks)

    assert headings == ["REAL SECTION HEADING"]
    # The dot-leader contents line is dropped entirely as navigation.
    assert "INTRODUCTION" not in text
    # Running page headers are demoted from headings (kept as plain text here).
    assert "4-111" not in {*headings}


def test_markdown_atx_heading_and_inline_table() -> None:
    md = "# Maintenance\n\nDo a check.\n\n| Item | Value |\n| --- | --- |\n| Torque | 40 Nm |\n"
    doc = TextDocumentParser().parse("note.md", md.encode("utf-8"))
    kinds = [b.kind for b in doc.blocks]
    assert BlockKind.HEADING in kinds
    assert BlockKind.TABLE in kinds
    heading = next(b for b in doc.blocks if b.kind == BlockKind.HEADING)
    assert heading.text == "Maintenance"
