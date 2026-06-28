# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Structure-aware chunking.

A naive splitter breaks tables across rows and cuts procedures in half, which
destroys exactly the content a technician needs whole. This chunker respects
document structure: a table becomes a single chunk with its rows preserved, a
run of numbered steps stays together, and headings open a new chunk and travel
with the content beneath them as section context.

Chunk identifiers are content-derived, so re-ingesting an unchanged document
yields the same ids and citations remain stable across runs.
"""

from __future__ import annotations

import hashlib

from .code_index import CodeIndexer
from .config import ChunkingConfig
from .models import BlockKind, Chunk, Document, TextBlock


def chunk_document(
    document: Document,
    config: ChunkingConfig,
    indexer: CodeIndexer,
) -> list[Chunk]:
    """Split ``document`` into retrievable chunks under the given config."""

    chunks: list[Chunk] = []
    current_section: str | None = None
    buffer: list[TextBlock] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        _emit(chunks, document, buffer, current_section, indexer)
        buffer = []
        buffer_len = 0

    blocks = list(document.blocks)
    i = 0
    while i < len(blocks):
        block = blocks[i]

        if block.kind == BlockKind.HEADING:
            flush()
            current_section = block.text
            i += 1
            continue

        if block.kind == BlockKind.TABLE and config.keep_tables_intact:
            flush()
            _emit(chunks, document, [block], current_section, indexer)
            i += 1
            continue

        if block.kind == BlockKind.STEP and config.keep_steps_intact:
            flush()
            run: list[TextBlock] = []
            while i < len(blocks) and blocks[i].kind == BlockKind.STEP:
                run.append(blocks[i])
                i += 1
            # Keep the procedure together, but divide an oversized run at step
            # boundaries so no single step is ever split.
            for group in _pack_blocks(run, config.step_max_chars):
                _emit(chunks, document, group, current_section, indexer)
            continue

        # Paragraphs and captions pack greedily up to the size budget.
        block_len = len(block.text)
        if buffer and buffer_len + block_len > config.max_chars:
            flush()
        buffer.append(block)
        buffer_len += block_len
        i += 1

    flush()
    return chunks


def _emit(
    chunks: list[Chunk],
    document: Document,
    blocks: list[TextBlock],
    section: str | None,
    indexer: CodeIndexer,
) -> None:
    text = "\n".join(block.text for block in blocks).strip()
    if not text:
        return
    page = blocks[0].page
    block_section = blocks[0].section if blocks[0].section is not None else section
    kind = _dominant_kind(blocks)
    rows = blocks[0].rows if kind == BlockKind.TABLE else None
    # Codes are detected on the section-prefixed text so that an identifier named
    # only in a heading still attaches to the body chunk beneath it.
    scan_text = f"{block_section}\n{text}" if block_section else text
    codes = indexer.extract(scan_text)
    chunk_id = _make_id(document.document_id, len(chunks), page, text)
    chunks.append(
        Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            source_path=document.source_path,
            text=text,
            page=page,
            section=block_section,
            kind=kind,
            codes=codes,
            rows=rows,
        )
    )


def _pack_blocks(blocks: list[TextBlock], max_chars: int) -> list[list[TextBlock]]:
    """Greedily group whole blocks into runs that stay within a size budget.

    A block is never split; if a single block already exceeds the budget it
    becomes its own group.
    """

    groups: list[list[TextBlock]] = []
    current: list[TextBlock] = []
    length = 0
    for block in blocks:
        block_len = len(block.text)
        if current and length + block_len > max_chars:
            groups.append(current)
            current = []
            length = 0
        current.append(block)
        length += block_len
    if current:
        groups.append(current)
    return groups


def _dominant_kind(blocks: list[TextBlock]) -> BlockKind:
    if any(b.kind == BlockKind.TABLE for b in blocks):
        return BlockKind.TABLE
    if blocks and all(b.kind == BlockKind.STEP for b in blocks):
        return BlockKind.STEP
    return BlockKind.PARAGRAPH


def _make_id(document_id: str, ordinal: int, page: int, text: str) -> str:
    digest = hashlib.sha1(
        f"{document_id}|{ordinal}|{page}|{text}".encode()
    ).hexdigest()
    return digest[:16]


def embedding_text(chunk: Chunk) -> str:
    """Return the text used to embed a chunk.

    The section heading is prepended so semantically thin chunks (for example a
    bare parameter table) still carry the context needed to be retrievable.
    """

    if chunk.section:
        return f"{chunk.section}\n{chunk.text}"
    return chunk.text
