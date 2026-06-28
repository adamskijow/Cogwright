# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Immutable data types shared across the core library.

These types form the contract between pipeline stages. They are deliberately
plain and serialization-friendly so that an index built on one machine can be
persisted and reopened on another without carrying any model or vendor state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

# A vector is a plain list of floats. The core never assumes a specific
# embedding dimension or backend; it only does arithmetic over these.
Vector = list[float]


class BlockKind(StrEnum):
    """The structural role a block of text plays in a source document."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    STEP = "step"
    TABLE = "table"
    CAPTION = "caption"


@dataclass(frozen=True)
class CodeRef:
    """A detected alarm, stop, fault, error, or part identifier.

    ``value`` is the normalized lookup key (for example ``AL-204``). ``raw`` is
    the text as it appeared in the source so callers can show it verbatim.
    """

    kind: str
    value: str
    raw: str


@dataclass(frozen=True)
class TextBlock:
    """A normalized unit of a parsed document.

    For tables, ``text`` holds a readable rendering and ``rows`` holds the
    structured cells so a later stage can keep them intact.
    """

    kind: BlockKind
    text: str
    page: int
    section: str | None = None
    rows: tuple[tuple[str, ...], ...] | None = None


@dataclass(frozen=True)
class Document:
    """A parsed source document, normalized into ordered blocks."""

    document_id: str
    source_path: str
    title: str
    blocks: tuple[TextBlock, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit with stable identity and a path back to its source."""

    chunk_id: str
    document_id: str
    source_path: str
    text: str
    page: int
    section: str | None
    kind: BlockKind
    codes: tuple[CodeRef, ...] = ()
    rows: tuple[tuple[str, ...], ...] | None = None


@dataclass(frozen=True)
class Message:
    """A single chat message handed to an ``LLMClient``."""

    role: str
    content: str


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk paired with its retrieval score and how it matched."""

    chunk: Chunk
    score: float
    match_type: str  # "semantic", "code", or "hybrid"


@dataclass(frozen=True)
class Citation:
    """A pointer from an answer back to a specific source location."""

    chunk_id: str
    source_path: str
    page: int
    section: str | None


@dataclass(frozen=True)
class Answer:
    """The structured result of a query.

    ``found`` is ``False`` when the question could not be grounded in the corpus.
    In that case ``text`` carries the not-found message and the other collections
    are empty.
    """

    text: str
    found: bool
    citations: tuple[Citation, ...] = ()
    referenced_codes: tuple[CodeRef, ...] = ()
    retrieved: tuple[ScoredChunk, ...] = ()
