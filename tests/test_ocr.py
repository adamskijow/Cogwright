# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for the scanned-page OCR routing in the PDF parser.

A PDF with no text layer is generated, then parsed with a fake OCR engine so the
routing decision and the render-then-recognize path are exercised end to end
without depending on a real recognition library.
"""

from __future__ import annotations

import io

import pytest

from cogwright.adapters.pdf_parser import PdfDocumentParser
from cogwright.core.chunking import chunk_document
from cogwright.core.code_index import CodeIndexer
from cogwright.core.config import DEFAULT_CODE_PATTERNS, ChunkingConfig
from cogwright.core.models import BlockKind

from .fakes import FakeOcrEngine

pytest.importorskip("reportlab")

OCR_TEXT = """ALARM REFERENCE

Alarm 204 indicates low coolant pressure.

1. Stop the unit and let it cool.
2. Refill coolant to the cold mark.
"""


def _image_only_pdf() -> bytes:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    # Draw only shapes, never text, so the page has no recoverable text layer.
    pdf.rect(72, 600, 200, 80, fill=1)
    pdf.circle(300, 400, 50, fill=1)
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _text_pdf() -> bytes:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    pdf.drawString(72, 700, "Born-digital page with a real text layer.")
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_scanned_page_is_routed_through_ocr() -> None:
    engine = FakeOcrEngine(OCR_TEXT)
    parser = PdfDocumentParser(ocr_engine=engine)

    doc = parser.parse("scan.pdf", _image_only_pdf())

    # The engine was called once with non-empty rendered image bytes.
    assert len(engine.calls) == 1
    assert engine.calls[0][:8] == b"\x89PNG\r\n\x1a\n"
    # The recognized text was structured like any other page.
    headings = {b.text for b in doc.blocks if b.kind == BlockKind.HEADING}
    assert "ALARM REFERENCE" in headings
    steps = [b for b in doc.blocks if b.kind == BlockKind.STEP]
    assert len(steps) == 2


def test_recognized_codes_resolve_after_chunking() -> None:
    parser = PdfDocumentParser(ocr_engine=FakeOcrEngine(OCR_TEXT))
    doc = parser.parse("scan.pdf", _image_only_pdf())
    chunks = chunk_document(doc, ChunkingConfig(), CodeIndexer(DEFAULT_CODE_PATTERNS))
    values = {code.value for chunk in chunks for code in chunk.codes}
    assert "AL-204" in values


def test_born_digital_page_does_not_call_ocr() -> None:
    engine = FakeOcrEngine("should not be used")
    parser = PdfDocumentParser(ocr_engine=engine)

    doc = parser.parse("text.pdf", _text_pdf())

    # The page already has text, so OCR is never invoked.
    assert engine.calls == []
    assert any("Born-digital" in b.text for b in doc.blocks)


def test_no_engine_means_scanned_pages_yield_no_blocks() -> None:
    parser = PdfDocumentParser()
    doc = parser.parse("scan.pdf", _image_only_pdf())
    assert doc.blocks == ()
