# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests that chunking respects document structure."""

from __future__ import annotations

from cogwright.core.chunking import chunk_document, embedding_text
from cogwright.core.code_index import CodeIndexer
from cogwright.core.config import DEFAULT_CODE_PATTERNS, ChunkingConfig
from cogwright.core.models import BlockKind

from .builders import block, document

INDEXER = CodeIndexer(DEFAULT_CODE_PATTERNS)


def test_table_stays_a_single_intact_chunk() -> None:
    rows = (
        ("Part", "Part Number"),
        ("Drive belt", "PN 44-19A"),
        ("Seal kit", "PN 7788-01"),
    )
    doc = document(
        "doc",
        block(BlockKind.HEADING, "REPLACEMENT PARTS", page=4),
        block(BlockKind.TABLE, "Part | Part Number\nDrive belt | PN 44-19A", page=4, rows=rows),
    )

    chunks = chunk_document(doc, ChunkingConfig(), INDEXER)

    table_chunks = [c for c in chunks if c.kind == BlockKind.TABLE]
    assert len(table_chunks) == 1
    table = table_chunks[0]
    assert table.rows == rows
    assert table.section == "REPLACEMENT PARTS"
    assert table.page == 4


def test_procedure_steps_are_kept_together() -> None:
    doc = document(
        "doc",
        block(BlockKind.HEADING, "STARTUP PROCEDURE", page=2),
        block(BlockKind.STEP, "1. Engage the main disconnect.", page=2),
        block(BlockKind.STEP, "2. Confirm the guard is closed.", page=2),
        block(BlockKind.STEP, "3. Press start and watch for motion.", page=2),
    )

    chunks = chunk_document(doc, ChunkingConfig(max_chars=20), INDEXER)

    step_chunks = [c for c in chunks if c.kind == BlockKind.STEP]
    # Even with a tiny size budget the run of steps is never split.
    assert len(step_chunks) == 1
    text = step_chunks[0].text
    assert "1. Engage" in text
    assert "2. Confirm" in text
    assert "3. Press start" in text
    assert step_chunks[0].section == "STARTUP PROCEDURE"


def test_oversized_step_run_splits_at_step_boundaries() -> None:
    steps = [
        block(BlockKind.STEP, f"{n}. " + ("x" * 40), page=2) for n in range(1, 7)
    ]
    doc = document("doc", block(BlockKind.HEADING, "LONG PROCEDURE", page=2), *steps)

    # Budget fits about two ~43-char steps per chunk, so the run divides.
    chunks = chunk_document(doc, ChunkingConfig(step_max_chars=90), INDEXER)

    step_chunks = [c for c in chunks if c.kind == BlockKind.STEP]
    assert len(step_chunks) > 1
    # No step is ever split: every original step line lands intact in some chunk.
    joined = "\n".join(c.text for c in step_chunks)
    for n in range(1, 7):
        assert f"{n}. " in joined
    assert all(c.section == "LONG PROCEDURE" for c in step_chunks)


def test_paragraphs_pack_up_to_the_size_budget() -> None:
    doc = document(
        "doc",
        block(BlockKind.HEADING, "OVERVIEW", page=1),
        block(BlockKind.PARAGRAPH, "A" * 30, page=1),
        block(BlockKind.PARAGRAPH, "B" * 30, page=1),
        block(BlockKind.PARAGRAPH, "C" * 30, page=1),
    )

    chunks = chunk_document(doc, ChunkingConfig(max_chars=50), INDEXER)

    # Three 30-char paragraphs cannot all fit in a 50-char budget, so they split.
    assert len(chunks) >= 2
    assert all(c.section == "OVERVIEW" for c in chunks)


def test_chunk_ids_are_stable_across_runs() -> None:
    doc = document(
        "doc",
        block(BlockKind.HEADING, "SECTION", page=1),
        block(BlockKind.PARAGRAPH, "Stable content here.", page=1),
    )

    first = chunk_document(doc, ChunkingConfig(), INDEXER)
    second = chunk_document(doc, ChunkingConfig(), INDEXER)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_codes_in_chunk_are_detected() -> None:
    doc = document(
        "doc",
        block(BlockKind.HEADING, "ALARM REFERENCE", page=3),
        block(BlockKind.PARAGRAPH, "Alarm 204 indicates low coolant.", page=3),
    )

    chunks = chunk_document(doc, ChunkingConfig(), INDEXER)
    body = next(c for c in chunks if "204" in c.text)
    assert "AL-204" in {code.value for code in body.codes}


def test_embedding_text_prepends_section() -> None:
    doc = document(
        "doc",
        block(BlockKind.HEADING, "COOLANT SYSTEM", page=1),
        block(BlockKind.PARAGRAPH, "Top up to the cold line.", page=1),
    )
    chunk = chunk_document(doc, ChunkingConfig(), INDEXER)[0]
    assert embedding_text(chunk).startswith("COOLANT SYSTEM")
