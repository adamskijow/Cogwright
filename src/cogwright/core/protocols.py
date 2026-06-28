# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Protocol seams that isolate the pure core from real input and output.

Every side effect the engine needs (reading files, parsing documents, embedding
text, calling a model, storing vectors) is expressed as a ``Protocol`` here. The
core depends only on these abstractions; the adapter layer supplies concrete
implementations and the test suite supplies fakes.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol, runtime_checkable

from .models import Document, Message, Vector


@runtime_checkable
class FileSystem(Protocol):
    """Abstracts file access so the core never touches the disk directly."""

    def read_bytes(self, path: str) -> bytes: ...

    def read_text(self, path: str) -> str: ...

    def write_text(self, path: str, data: str) -> None: ...

    def exists(self, path: str) -> bool: ...

    def list_files(self, path: str, extensions: Sequence[str]) -> list[str]: ...


@runtime_checkable
class DocumentParser(Protocol):
    """Turns raw document bytes into a normalized :class:`Document`.

    Implementations declare which paths they handle through :meth:`supports`,
    which lets the ingestion pipeline route each file to the right parser and
    lets future parsers (scanned pages, diagrams) slot in without core changes.
    """

    def supports(self, path: str) -> bool: ...

    def parse(self, path: str, data: bytes) -> Document: ...


@runtime_checkable
class Embedder(Protocol):
    """Maps text to vectors. Batched so backends can amortize round trips."""

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


@runtime_checkable
class LLMClient(Protocol):
    """A chat-style language model behind a uniform, vendor-neutral surface."""

    def complete(self, messages: Sequence[Message]) -> str: ...

    def stream(self, messages: Sequence[Message]) -> Iterator[str]: ...

    def available(self) -> bool: ...


@runtime_checkable
class VectorStore(Protocol):
    """Holds chunk vectors and answers nearest-neighbor queries.

    The reference implementation keeps everything in memory and scores with
    cosine similarity, but the seam allows swapping in an approximate index later
    without touching retrieval logic.
    """

    def add(self, chunk_id: str, vector: Vector) -> None: ...

    def search(self, vector: Vector, k: int) -> list[tuple[str, float]]: ...

    def vectors(self) -> Mapping[str, Vector]: ...

    def load(self, data: Mapping[str, Vector]) -> None: ...


@runtime_checkable
class OcrEngine(Protocol):
    """Recognizes text in a rendered page image.

    This is the seam for milestone-two scanned-page support. A PDF page that has
    no recoverable text layer is rendered to an image and handed here; the parser
    then structures the returned text exactly as it does born-digital text. The
    engine receives encoded image bytes (PNG) so the core stays free of any
    imaging or model dependency.
    """

    def image_to_text(self, image: bytes) -> str: ...


@runtime_checkable
class DiagramAnalyzer(Protocol):
    """Describes the figures and callouts on a rendered page image.

    This is the seam for exploded-diagram and callout understanding. A page that
    contains figures is rendered and handed here; the returned strings, one per
    callout or caption, are attached as caption blocks so a query can reach the
    labels printed on a diagram. The analyzer owns any region detection it needs
    and receives encoded image bytes (PNG), keeping the core imaging-free.
    """

    def describe(self, image: bytes) -> Sequence[str]: ...
