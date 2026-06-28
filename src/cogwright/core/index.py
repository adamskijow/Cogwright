# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""The persisted retrieval index and its serialization.

An :class:`Index` bundles the three things a query needs: the chunks (for text
and citation), a vector store (for semantic search), and a code index (for exact
identifier lookup). Serialization produces plain JSON-compatible structures so an
index is portable and contains no model or vendor state. The code index is
rebuilt from the chunks on load rather than stored, keeping the format small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .code_index import CodeIndex
from .models import BlockKind, Chunk, CodeRef
from .protocols import VectorStore
from .vector_store import InMemoryVectorStore

SCHEMA_VERSION = 1


@dataclass
class Index:
    """A queryable, serializable index over a document corpus."""

    chunks: dict[str, Chunk]
    store: VectorStore
    code_index: CodeIndex

    @classmethod
    def build(cls, chunks: list[Chunk], store: VectorStore) -> Index:
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        return cls(
            chunks=chunk_map,
            store=store,
            code_index=CodeIndex.from_chunks(chunks),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "chunks": [_chunk_to_dict(c) for c in self.chunks.values()],
            "vectors": {cid: list(vec) for cid, vec in self.store.vectors().items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Index:
        version = data.get("version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported index version {version!r}; expected {SCHEMA_VERSION}"
            )
        chunks = [_chunk_from_dict(c) for c in data["chunks"]]
        store: VectorStore = InMemoryVectorStore()
        store.load({cid: list(vec) for cid, vec in data["vectors"].items()})
        index = cls.build(chunks, store)
        return index


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_path": chunk.source_path,
        "text": chunk.text,
        "page": chunk.page,
        "section": chunk.section,
        "kind": chunk.kind.value,
        "codes": [
            {"kind": c.kind, "value": c.value, "raw": c.raw} for c in chunk.codes
        ],
        "rows": [list(row) for row in chunk.rows] if chunk.rows is not None else None,
    }


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    rows_data = data.get("rows")
    rows = (
        tuple(tuple(str(cell) for cell in row) for row in rows_data)
        if rows_data is not None
        else None
    )
    return Chunk(
        chunk_id=data["chunk_id"],
        document_id=data["document_id"],
        source_path=data["source_path"],
        text=data["text"],
        page=int(data["page"]),
        section=data["section"],
        kind=BlockKind(data["kind"]),
        codes=tuple(
            CodeRef(kind=c["kind"], value=c["value"], raw=c["raw"])
            for c in data.get("codes", ())
        ),
        rows=rows,
    )
