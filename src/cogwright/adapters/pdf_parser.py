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
import re
from collections import Counter
from typing import Any

from ..core.errors import CogwrightError
from ..core.models import BlockKind, Document, TextBlock
from ..core.protocols import DiagramAnalyzer, OcrEngine
from .text_parser import parse_lines

# A line counts as a running header or footer when its digit-normalized form
# repeats on at least this fraction of pages (with a small floor for short docs).
_HEADER_FOOTER_FRACTION = 10
_HEADER_FOOTER_MIN_PAGES = 4


class PdfDocumentParser:
    """Parses ``.pdf`` documents into normalized blocks, page by page."""

    def __init__(
        self,
        ocr_engine: OcrEngine | None = None,
        ocr_dpi: int = 200,
        ocr_min_chars: int = 16,
        ocr_image_ratio: float = 0.5,
        diagram_analyzer: DiagramAnalyzer | None = None,
        diagram_min_image_ratio: float = 0.1,
    ) -> None:
        # When no engine is supplied, scanned pages are skipped rather than
        # guessed at, keeping born-digital behavior identical to before.
        self._ocr_engine = ocr_engine
        self._ocr_dpi = ocr_dpi
        # A page is treated as scanned when its recoverable text is shorter than
        # this and a raster image covers at least this fraction of the page. The
        # text threshold (rather than strictly empty) catches pages that carry
        # only a stray page number; the image-coverage test keeps born-digital
        # pages, including table-only ones drawn with vector rules, from being
        # needlessly rerecognized.
        self._ocr_min_chars = ocr_min_chars
        self._ocr_image_ratio = ocr_image_ratio
        # A page is analyzed for diagram callouts when an embedded image covers
        # at least this fraction of it, so a small logo does not trigger work.
        self._diagram_analyzer = diagram_analyzer
        self._diagram_min_image_ratio = diagram_min_image_ratio

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
            # First pass: pull the text and tables from every page. Second pass
            # builds blocks after the running headers and footers that repeat
            # across pages have been identified and removed.
            pages = [
                (page, number, *self._extract_text_and_tables(page))
                for number, page in enumerate(pdf.pages, start=1)
            ]
            repeating = _repeating_header_footer([text for _, _, text, _ in pages])
            for page, number, text, tables in pages:
                cleaned = _strip_repeating(text, repeating)
                for block in self._build_blocks(page, number, cleaned, tables):
                    if title is None and block.kind == BlockKind.HEADING:
                        title = block.text
                    blocks.append(block)

        return Document(
            document_id=stem,
            source_path=path,
            title=title or stem,
            blocks=tuple(blocks),
        )

    def _extract_text_and_tables(self, page: Any) -> tuple[str, list[Any]]:
        tables = page.find_tables()
        bboxes = [table.bbox for table in tables]
        # Extract the running text from the part of the page that is not inside a
        # table, so table cells are not also chunked as paragraphs. Without this,
        # the same table content would be indexed twice.
        if bboxes:
            text = page.filter(lambda obj: not _inside_any(obj, bboxes)).extract_text()
        else:
            text = page.extract_text()
        return text or "", list(tables)

    def _build_blocks(
        self, page: Any, page_number: int, text: str, tables: list[Any]
    ) -> list[TextBlock]:
        image_ratio = _image_area_ratio(page)
        # The rendered page image is shared by the OCR and diagram passes so a
        # page that needs both is only rasterized once.
        rendered: bytes | None = None

        if self._ocr_engine is not None and self._needs_ocr(text, image_ratio):
            rendered = _render_png(page, self._ocr_dpi)
            text = self._ocr_engine.image_to_text(rendered)

        blocks = parse_lines(text, page_number)
        for table in tables:
            table_block = _table_block(table.extract(), page_number)
            if table_block is not None:
                blocks.append(table_block)

        if (
            self._diagram_analyzer is not None
            and image_ratio >= self._diagram_min_image_ratio
        ):
            if rendered is None:
                rendered = _render_png(page, self._ocr_dpi)
            for caption in self._diagram_analyzer.describe(rendered):
                text_value = caption.strip()
                if text_value:
                    blocks.append(
                        TextBlock(
                            kind=BlockKind.CAPTION,
                            text=text_value,
                            page=page_number,
                            section="Figure",
                        )
                    )
        return blocks

    def _needs_ocr(self, text: str, image_ratio: float) -> bool:
        if len(text.strip()) >= self._ocr_min_chars:
            return False
        return image_ratio >= self._ocr_image_ratio


def _normalize_line(line: str) -> str:
    """Collapse digits so "EDITOR 4-111" and "EDITOR 6-205" compare as equal."""

    return re.sub(r"\d+", "#", line.strip())


def _repeating_header_footer(texts: list[str]) -> frozenset[str]:
    """Find the normalized first and last lines that repeat across many pages.

    Running headers and footers (chapter titles, page numbers, revision dates)
    sit at the top or bottom of most pages and add noise to every chunk. A line
    is treated as one when its digit-normalized form recurs widely.
    """

    first: Counter[str] = Counter()
    last: Counter[str] = Counter()
    for text in texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        first[_normalize_line(lines[0])] += 1
        if len(lines) > 1:
            last[_normalize_line(lines[-1])] += 1

    threshold = max(_HEADER_FOOTER_MIN_PAGES, len(texts) // _HEADER_FOOTER_FRACTION)
    repeating = {norm for norm, n in first.items() if norm and n >= threshold}
    repeating |= {norm for norm, n in last.items() if norm and n >= threshold}
    return frozenset(repeating)


def _strip_repeating(text: str, repeating: frozenset[str]) -> str:
    """Remove the leading and trailing line when it is a known header or footer."""

    if not repeating:
        return text
    lines = text.splitlines()
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    if not nonempty:
        return text
    drop: set[int] = set()
    if _normalize_line(lines[nonempty[0]]) in repeating:
        drop.add(nonempty[0])
    if len(nonempty) > 1 and _normalize_line(lines[nonempty[-1]]) in repeating:
        drop.add(nonempty[-1])
    if not drop:
        return text
    return "\n".join(line for i, line in enumerate(lines) if i not in drop)


def _image_area_ratio(page: Any) -> float:
    """Fraction of the page covered by embedded raster images, capped at 1.0."""

    page_area = float(page.width) * float(page.height)
    if page_area <= 0:
        return 0.0
    covered = 0.0
    for image in page.images:
        covered += (image["x1"] - image["x0"]) * (image["bottom"] - image["top"])
    return min(covered / page_area, 1.0)


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
