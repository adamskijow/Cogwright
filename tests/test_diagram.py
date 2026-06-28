# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for the diagram-region analysis seam in the PDF parser.

A PDF with an embedded figure and ordinary text is parsed with a fake analyzer,
so the figure-detection routing and the caption blocks it produces are exercised
without depending on a vision model.
"""

from __future__ import annotations

import io

import pytest

from cogwright.adapters.pdf_parser import PdfDocumentParser
from cogwright.core.chunking import chunk_document
from cogwright.core.code_index import CodeIndexer
from cogwright.core.config import DEFAULT_CODE_PATTERNS, ChunkingConfig
from cogwright.core.models import BlockKind

from .fakes import FakeDiagramAnalyzer

pytest.importorskip("reportlab")

CAPTIONS = [
    "Figure 1: callout A points to the drive belt, part PN 44-19A.",
    "Figure 1: callout B points to the tensioner.",
]


def _figure_pdf() -> bytes:
    from PIL import Image
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    figure = Image.new("RGB", (400, 400), (180, 180, 180))
    figure_bytes = io.BytesIO()
    figure.save(figure_bytes, format="PNG")
    figure_bytes.seek(0)

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    pdf.drawString(72, 740, "DRIVE ASSEMBLY")
    pdf.drawString(72, 720, "The drive assembly is shown in the figure below.")
    # A figure covering enough of the page to be analyzed, but with the text
    # above it intact so the page is not treated as scanned.
    pdf.drawImage(ImageReader(figure_bytes), 100, 200, width=300, height=300)
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_figure_page_produces_caption_blocks() -> None:
    analyzer = FakeDiagramAnalyzer(CAPTIONS)
    parser = PdfDocumentParser(diagram_analyzer=analyzer)

    doc = parser.parse("guide.pdf", _figure_pdf())

    assert len(analyzer.calls) == 1
    assert analyzer.calls[0][:8] == b"\x89PNG\r\n\x1a\n"
    captions = [b for b in doc.blocks if b.kind == BlockKind.CAPTION]
    assert len(captions) == 2
    assert all(b.section == "Figure" for b in captions)
    # The born-digital text is still present alongside the captions.
    assert any("drive assembly" in b.text.lower() for b in doc.blocks)


def test_callout_identifiers_resolve_after_chunking() -> None:
    parser = PdfDocumentParser(diagram_analyzer=FakeDiagramAnalyzer(CAPTIONS))
    doc = parser.parse("guide.pdf", _figure_pdf())
    chunks = chunk_document(doc, ChunkingConfig(), CodeIndexer(DEFAULT_CODE_PATTERNS))
    values = {code.value for chunk in chunks for code in chunk.codes}
    # A part number printed as a diagram callout becomes a resolvable identifier.
    assert "PN-44-19A" in values


def test_no_analyzer_means_no_caption_blocks() -> None:
    parser = PdfDocumentParser()
    doc = parser.parse("guide.pdf", _figure_pdf())
    assert not any(b.kind == BlockKind.CAPTION for b in doc.blocks)
