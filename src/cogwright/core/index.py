# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""The persisted retrieval index, its metadata, and its serialization.

An :class:`Index` bundles the chunks (for text and citation), a vector store
(for semantic search), and a code index (for exact identifier lookup), along with
:class:`IndexMetadata` describing how and when it was built. The metadata records
which documents are present and the embedding model that produced the vectors, so
the lifecycle commands can add, refresh, and remove documents and so a query can
warn when it is run with a different embedding model than the index was built
with. Serialization produces plain JSON-compatible structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .code_index import CodeIndex
from .models import BlockKind, Chunk, CodeRef, Vector
from .protocols import VectorStore
from .vector_store import InMemoryVectorStore

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DocumentRecord:
    """What the index knows about one ingested document."""

    source_path: str
    document_id: str
    title: str
    content_hash: str
    chunk_count: int


@dataclass(frozen=True)
class IndexMetadata:
    """Provenance for an index: how, when, and over what it was built."""

    embedding_model: str = ""
    vector_dim: int | None = None
    created_at: str = ""
    updated_at: str = ""
    documents: tuple[DocumentRecord, ...] = ()

    def document(self, source_path: str) -> DocumentRecord | None:
        return next((d for d in self.documents if d.source_path == source_path), None)


@dataclass
class Index:
    """A queryable, serializable index over a document corpus."""

    chunks: dict[str, Chunk]
    store: VectorStore
    code_index: CodeIndex
    metadata: IndexMetadata = field(default_factory=IndexMetadata)

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        store: VectorStore,
        metadata: IndexMetadata | None = None,
    ) -> Index:
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        return cls(
            chunks=chunk_map,
            store=store,
            code_index=CodeIndex.from_chunks(chunks),
            metadata=metadata or IndexMetadata(),
        )

    def add_chunks(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        """Add a document's chunks and their vectors to the index."""

        for chunk, vector in zip(chunks, vectors, strict=True):
            self.chunks[chunk.chunk_id] = chunk
            self.store.add(chunk.chunk_id, vector)
            self.code_index.add_chunk(chunk)

    def remove_document(self, source_path: str) -> int:
        """Drop every chunk that came from ``source_path``; return how many."""

        ids = [
            cid for cid, chunk in self.chunks.items() if chunk.source_path == source_path
        ]
        for cid in ids:
            del self.chunks[cid]
            self.store.remove(cid)
        # The code index has no per-entry removal, so rebuild it from what remains.
        self.code_index = CodeIndex.from_chunks(self.chunks.values())
        return len(ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "metadata": _metadata_to_dict(self.metadata),
            "chunks": [_chunk_to_dict(c) for c in self.chunks.values()],
            "vectors": {cid: list(vec) for cid, vec in self.store.vectors().items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Index:
        version = data.get("version")
        if version not in (1, SCHEMA_VERSION):
            raise ValueError(
                f"unsupported index version {version!r}; expected 1 or {SCHEMA_VERSION}"
            )
        chunks = [_chunk_from_dict(c) for c in data["chunks"]]
        store: VectorStore = InMemoryVectorStore()
        store.load({cid: list(vec) for cid, vec in data["vectors"].items()})
        return cls.build(chunks, store, _metadata_from_dict(data.get("metadata")))


def _metadata_to_dict(metadata: IndexMetadata) -> dict[str, Any]:
    return {
        "embedding_model": metadata.embedding_model,
        "vector_dim": metadata.vector_dim,
        "created_at": metadata.created_at,
        "updated_at": metadata.updated_at,
        "documents": [
            {
                "source_path": d.source_path,
                "document_id": d.document_id,
                "title": d.title,
                "content_hash": d.content_hash,
                "chunk_count": d.chunk_count,
            }
            for d in metadata.documents
        ],
    }


def _metadata_from_dict(data: dict[str, Any] | None) -> IndexMetadata:
    if not data:
        return IndexMetadata()
    documents = tuple(
        DocumentRecord(
            source_path=record["source_path"],
            document_id=record.get("document_id", ""),
            title=record.get("title", ""),
            content_hash=record.get("content_hash", ""),
            chunk_count=int(record.get("chunk_count", 0)),
        )
        for record in data.get("documents", ())
    )
    return IndexMetadata(
        embedding_model=data.get("embedding_model", ""),
        vector_dim=data.get("vector_dim"),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        documents=documents,
    )


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
