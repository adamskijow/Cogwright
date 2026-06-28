# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Small constructors shared across tests to keep cases readable."""

from __future__ import annotations

from cogwright.core.models import (
    BlockKind,
    Chunk,
    CodeRef,
    Document,
    TextBlock,
    Vector,
)
from cogwright.core.vector_store import InMemoryVectorStore
from cogwright.core.index import Index


def block(
    kind: BlockKind,
    text: str,
    page: int = 1,
    section: str | None = None,
    rows: tuple[tuple[str, ...], ...] | None = None,
) -> TextBlock:
    return TextBlock(kind=kind, text=text, page=page, section=section, rows=rows)


def document(document_id: str, *blocks: TextBlock, path: str = "manual.txt") -> Document:
    return Document(
        document_id=document_id,
        source_path=path,
        title=document_id,
        blocks=tuple(blocks),
    )


def chunk(
    chunk_id: str,
    text: str = "body",
    page: int = 1,
    section: str | None = None,
    kind: BlockKind = BlockKind.PARAGRAPH,
    codes: tuple[CodeRef, ...] = (),
    source_path: str = "manual.txt",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        source_path=source_path,
        text=text,
        page=page,
        section=section,
        kind=kind,
        codes=codes,
    )


def index_with(chunk_vectors: dict[Chunk, Vector]) -> Index:
    store = InMemoryVectorStore()
    chunks = list(chunk_vectors)
    for ch, vec in chunk_vectors.items():
        store.add(ch.chunk_id, vec)
    return Index.build(chunks, store)
