# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for the born-digital PDF parser against real PDF bytes.

A small PDF is generated in-process with a dev-only toolkit, so the parser is
exercised on genuine page layout: a heading, body text, and a ruled table across
two pages.
"""

from __future__ import annotations

import io

import pytest

from cogwright.adapters.pdf_parser import PdfDocumentParser
from cogwright.core.chunking import chunk_document
from cogwright.core.code_index import CodeIndexer
from cogwright.core.config import DEFAULT_CODE_PATTERNS, ChunkingConfig
from cogwright.core.models import BlockKind, Document

pytest.importorskip("reportlab")


def _make_pdf() -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER)
    styles = getSampleStyleSheet()
    grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)])
    story = [
        Paragraph("ALARM REFERENCE", styles["Heading1"]),
        Paragraph("Alarm 204 indicates low coolant pressure.", styles["BodyText"]),
        Spacer(1, 18),
        Table(
            [
                ["Component", "Part Number"],
                ["Drive belt", "PN 44-19A"],
                ["Seal kit", "PN 7788-01"],
            ],
            style=grid,
        ),
        PageBreak(),
        Paragraph("REPLACEMENT NOTES", styles["Heading1"]),
        Paragraph("Inspect the drive belt every 500 hours.", styles["BodyText"]),
    ]
    doc.build(story)
    return buffer.getvalue()


def _parse() -> Document:
    return PdfDocumentParser().parse("manual.pdf", _make_pdf())


def test_table_is_extracted_with_rows() -> None:
    doc = _parse()
    tables = [b for b in doc.blocks if b.kind == BlockKind.TABLE]
    assert len(tables) == 1
    rows = tables[0].rows
    assert rows is not None
    flattened = [cell for row in rows for cell in row]
    assert "PN 44-19A" in flattened
    assert "Drive belt" in flattened


def test_table_content_is_not_duplicated_into_paragraphs() -> None:
    doc = _parse()
    paragraphs = [b for b in doc.blocks if b.kind == BlockKind.PARAGRAPH]
    # The part number lives only in the structured table, never in running text.
    assert all("PN 44-19A" not in b.text for b in paragraphs)
    assert all("Drive belt" not in b.text for b in paragraphs)


def test_pages_and_headings_are_recovered() -> None:
    doc = _parse()
    assert max(b.page for b in doc.blocks) == 2
    headings = {b.text for b in doc.blocks if b.kind == BlockKind.HEADING}
    assert "ALARM REFERENCE" in headings
    # The second heading is on the second page.
    notes = next(b for b in doc.blocks if b.text == "REPLACEMENT NOTES")
    assert notes.page == 2


def test_codes_resolve_after_chunking_a_pdf() -> None:
    doc = _parse()
    chunks = chunk_document(doc, ChunkingConfig(), CodeIndexer(DEFAULT_CODE_PATTERNS))
    values = {code.value for chunk in chunks for code in chunk.codes}
    assert "AL-204" in values
    assert "PN-44-19A" in values
