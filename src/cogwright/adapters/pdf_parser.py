# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""A parser for born-digital PDF manuals, with an optional scanned-page path.

It uses a permissively licensed PDF toolkit to pull text and tables out of each
page, keeping real page numbers for citations and lifting tables out as
structured blocks so the chunker can keep them intact. The toolkit is imported
lazily, so text-only deployments never load it.

Born-digital pages with a recoverable text layer are the primary path. When a
page has no text layer and an :class:`OcrEngine` is supplied, the page is
rendered to an image and recognized through that seam, then structured exactly
like born-digital text. Without an engine, such a page simply yields no blocks,
so behavior is unchanged for text-only use. Diagram-region understanding remains
future work behind this same seam.
"""

from __future__ import annotations

import io
import os
from typing import Any

from ..core.errors import CogwrightError
from ..core.models import BlockKind, Document, TextBlock
from ..core.protocols import OcrEngine
from .text_parser import parse_lines


class PdfDocumentParser:
    """Parses ``.pdf`` documents into normalized blocks, page by page."""

    def __init__(
        self,
        ocr_engine: OcrEngine | None = None,
        ocr_dpi: int = 200,
    ) -> None:
        # When no engine is supplied, scanned pages are skipped rather than
        # guessed at, keeping born-digital behavior identical to before.
        self._ocr_engine = ocr_engine
        self._ocr_dpi = ocr_dpi

    def supports(self, path: str) -> bool:
        return path.lower().endswith(".pdf")

    def parse(self, path: str, data: bytes) -> Document:
        try:
            import pdfplumber
        except ImportError as exc:  # pragma: no cover - only without the dep
            raise CogwrightError(
                "PDF support requires the 'pdfplumber' package to be installed."
            ) from exc

        stem = os.path.splitext(os.path.basename(path))[0]
        blocks: list[TextBlock] = []
        title: str | None = None

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                for block in self._page_blocks(page, page_number):
                    if title is None and block.kind == BlockKind.HEADING:
                        title = block.text
                    blocks.append(block)

        return Document(
            document_id=stem,
            source_path=path,
            title=title or stem,
            blocks=tuple(blocks),
        )

    def _page_blocks(self, page: Any, page_number: int) -> list[TextBlock]:
        tables = page.find_tables()
        bboxes = [table.bbox for table in tables]

        # Extract the running text from the part of the page that is not inside a
        # table, so table cells are not also chunked as paragraphs. Without this,
        # the same table content would be indexed twice.
        if bboxes:
            text_page = page.filter(lambda obj: not _inside_any(obj, bboxes))
            text = text_page.extract_text() or ""
        else:
            text = page.extract_text() or ""

        # A page with no recoverable text layer is treated as scanned: render it
        # and recognize the text through the OCR seam when one is configured.
        if not text.strip() and self._ocr_engine is not None:
            text = self._ocr_engine.image_to_text(_render_png(page, self._ocr_dpi))

        blocks = parse_lines(text, page_number)
        for table in tables:
            table_block = _table_block(table.extract(), page_number)
            if table_block is not None:
                blocks.append(table_block)
        return blocks


def _render_png(page: Any, dpi: int) -> bytes:
    """Render a PDF page to PNG image bytes for recognition."""

    image = page.to_image(resolution=dpi)
    buffer = io.BytesIO()
    image.original.save(buffer, format="PNG")
    return buffer.getvalue()


def _inside_any(obj: Any, bboxes: list[Any]) -> bool:
    """Whether a page object's center falls within any table bounding box."""

    x = (obj["x0"] + obj["x1"]) / 2
    y = (obj["top"] + obj["bottom"]) / 2
    for x0, top, x1, bottom in bboxes:
        if x0 <= x <= x1 and top <= y <= bottom:
            return True
    return False


def _table_block(table: object, page: int) -> TextBlock | None:
    raw_rows = table if isinstance(table, list) else []
    rows: list[tuple[str, ...]] = []
    for raw in raw_rows:
        cells = tuple((cell or "").strip() for cell in raw)
        if any(cells):
            rows.append(cells)
    if not rows:
        return None
    text = "\n".join(" | ".join(row) for row in rows)
    return TextBlock(kind=BlockKind.TABLE, text=text, page=page, rows=tuple(rows))
