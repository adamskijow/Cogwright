# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Concrete implementations of the core protocol seams.

This layer is the only place that performs real input and output: reading files,
parsing PDFs, and talking to a configured endpoint. The pure core never imports
anything from here.
"""

from __future__ import annotations

from .filesystem import RealFileSystem
from .http_endpoint import HttpEmbedder, HttpLLMClient
from .ocr import PytesseractOcrEngine
from .pdf_parser import PdfDocumentParser
from .text_parser import TextDocumentParser

__all__ = [
    "HttpEmbedder",
    "HttpLLMClient",
    "PdfDocumentParser",
    "PytesseractOcrEngine",
    "RealFileSystem",
    "TextDocumentParser",
]
