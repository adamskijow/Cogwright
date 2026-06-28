# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for ingestion and the query engine wired with fakes."""

from __future__ import annotations

from cogwright.core.config import Config
from cogwright.core.engine import IngestionPipeline, QueryEngine
from cogwright.core.index import Index
from cogwright.core.models import BlockKind, Document
from cogwright.core.prompt import NOT_FOUND_MESSAGE

from .builders import block, document
from .fakes import FakeDocumentParser, FakeEmbedder, FakeFileSystem, FakeLLMClient

PATH = "corpus/manual.txt"


def _sample_document() -> Document:
    return document(
        "manual",
        block(BlockKind.HEADING, "ALARM REFERENCE", page=3),
        block(BlockKind.PARAGRAPH, "Alarm 204 indicates low coolant pressure.", page=3),
        block(BlockKind.STEP, "1. Stop the unit and let it cool.", page=3),
        block(BlockKind.STEP, "2. Refill coolant to the cold mark.", page=3),
        block(BlockKind.STEP, "3. Clear alarm 204 and restart the unit.", page=3),
        path=PATH,
    )


def _build_index(embedder: FakeEmbedder, config: Config) -> Index:
    fs = FakeFileSystem()
    fs.add_text(PATH, "ignored body; the fake parser supplies the document")
    parser = FakeDocumentParser({PATH: _sample_document()})
    pipeline = IngestionPipeline(fs, [parser], embedder, config)
    return pipeline.ingest([PATH])


def test_ingest_builds_chunks_vectors_and_code_index() -> None:
    embedder = FakeEmbedder()
    index = _build_index(embedder, Config())

    assert index.chunks
    # Every chunk has a vector in the store.
    assert set(index.store.vectors()) == set(index.chunks)
    # The alarm identifier is indexed for exact lookup.
    assert "AL-204" in index.code_index.values


def test_ask_returns_grounded_cited_step_answer() -> None:
    embedder = FakeEmbedder()
    config = Config()
    index = _build_index(embedder, config)

    llm = FakeLLMClient()
    engine = QueryEngine(embedder, llm, config)
    answer = engine.ask(index, "How do I clear alarm 204?")

    assert answer.found is True
    # Numbered procedure steps come through.
    assert "1." in answer.text and "2." in answer.text
    # The queried identifier is surfaced and resolved.
    assert "AL-204" in {code.value for code in answer.referenced_codes}
    # The answer is cited back to the source page that documents the alarm.
    assert answer.citations
    assert any(c.page == 3 for c in answer.citations)
    # The model was consulted exactly once.
    assert len(llm.calls) == 1


def test_not_found_path_skips_the_model() -> None:
    embedder = FakeEmbedder()
    config = Config()
    index = _build_index(embedder, config)

    llm = FakeLLMClient()
    engine = QueryEngine(embedder, llm, config)
    answer = engine.ask(index, "Bluetooth pairing instructions?")

    assert answer.found is False
    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.citations == ()
    # No hallucinated procedure: the model is never called when nothing is retrieved.
    assert llm.calls == []


def test_index_survives_a_serialization_round_trip() -> None:
    embedder = FakeEmbedder()
    config = Config()
    index = _build_index(embedder, config)

    restored = Index.from_dict(index.to_dict())

    assert set(restored.chunks) == set(index.chunks)
    assert restored.store.vectors() == index.store.vectors()
    assert restored.code_index.values == index.code_index.values

    # And a query against the restored index still works end to end.
    llm = FakeLLMClient()
    engine = QueryEngine(embedder, llm, config)
    answer = engine.ask(restored, "How do I clear alarm 204?")
    assert answer.found is True
    assert any(c.page == 3 for c in answer.citations)
