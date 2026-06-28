# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Validation of the real reference OCR engine.

These run only where the optical-recognition extra and the recognition binary
are both available, and are skipped otherwise, so the default suite and CI stay
green without them. They render text to an image and confirm the reference engine
reads it back, both directly and through the scanned-page path of the PDF parser.
"""

from __future__ import annotations

import io
import shutil
from typing import Any

import pytest

from cogwright.adapters.ocr import PytesseractOcrEngine
from cogwright.adapters.pdf_parser import PdfDocumentParser
from cogwright.core.chunking import chunk_document
from cogwright.core.code_index import CodeIndexer
from cogwright.core.config import DEFAULT_CODE_PATTERNS, ChunkingConfig
from cogwright.core.models import BlockKind

pytest.importorskip("pytesseract")
pytest.importorskip("reportlab")
pytest.importorskip("PIL")

if shutil.which("tesseract") is None:  # pragma: no cover - environment dependent
    pytest.skip("the tesseract binary is not installed", allow_module_level=True)


def _font(size: int) -> Any:
    from PIL import ImageFont

    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_image(lines: list[str]) -> Any:
    from PIL import Image, ImageDraw

    line_height = 70
    image = Image.new("RGB", (1000, line_height * (len(lines) + 2)), "white")
    draw = ImageDraw.Draw(image)
    font = _font(40)
    y = 30
    for line in lines:
        draw.text((40, y), line, fill="black", font=font)
        y += line_height
    return image


def test_engine_reads_rendered_text() -> None:
    image = _text_image(["Alarm 204 clears after refill"])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    recognized = PytesseractOcrEngine().image_to_text(buffer.getvalue())

    assert "204" in recognized
    assert "alarm" in recognized.lower()


def _scanned_manual_pdf() -> bytes:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    image = _text_image(
        [
            "ALARM REFERENCE",
            "",
            "Alarm 204 indicates low coolant pressure.",
            "",
            "1. Stop the unit and let it cool.",
            "2. Refill coolant to the cold mark.",
        ]
    )
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    pdf.drawImage(ImageReader(image_bytes), 20, 30, width=572, height=732)
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_scanned_pdf_is_recognized_and_codes_resolve() -> None:
    parser = PdfDocumentParser(ocr_engine=PytesseractOcrEngine())
    doc = parser.parse("scan.pdf", _scanned_manual_pdf())

    assert doc.blocks
    chunks = chunk_document(doc, ChunkingConfig(), CodeIndexer(DEFAULT_CODE_PATTERNS))
    values = {code.value for chunk in chunks for code in chunk.codes}
    assert "AL-204" in values
    # The all-caps section title is recovered as a heading.
    assert any(b.kind == BlockKind.HEADING for b in doc.blocks)
