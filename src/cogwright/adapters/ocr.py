# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""A reference OCR engine behind the :class:`OcrEngine` seam.

This wraps a permissively licensed optical-recognition library, imported lazily,
so the dependency is needed only when scanned-page support is actually used.
Install it with the optional extra (``cogwright[ocr]``). The seam means any other
engine can be substituted without changing the parser or the core.
"""

from __future__ import annotations

import io

from ..core.errors import CogwrightError


class PytesseractOcrEngine:
    """Recognizes page text using a Tesseract binding.

    The binding and image library are imported on first use. If they are not
    installed, a clear error explains how to enable scanned-page support.
    """

    def __init__(self, language: str = "eng") -> None:
        self._language = language

    def image_to_text(self, image: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise CogwrightError(
                "Scanned-page support requires the OCR extra. Install it with "
                "'cogwright[ocr]' and ensure the Tesseract engine is available."
            ) from exc

        page_image = Image.open(io.BytesIO(image))
        text: str = pytesseract.image_to_string(page_image, lang=self._language)
        return text
