# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for incremental index updates, removal, and metadata."""

from __future__ import annotations

import pytest

from cogwright.adapters.text_parser import TextDocumentParser
from cogwright.core.config import Config
from cogwright.core.engine import IngestionPipeline
from cogwright.core.errors import CogwrightError
from cogwright.core.index import Index

from .fakes import FakeEmbedder, FakeFileSystem


def _pipeline(fs: FakeFileSystem, embedder: FakeEmbedder) -> IngestionPipeline:
    return IngestionPipeline(fs, [TextDocumentParser()], embedder, Config())


def _corpus() -> FakeFileSystem:
    fs = FakeFileSystem()
    fs.add_text("a.txt", "ALARM REFERENCE\n\nAlarm 204 means low coolant.\n")
    fs.add_text("b.txt", "STARTUP\n\n1. Turn the unit on.\n")
    return fs


def test_ingest_stamps_metadata() -> None:
    fs = _corpus()
    pipeline = _pipeline(fs, FakeEmbedder())
    index = pipeline.ingest(["a.txt", "b.txt"], embedding_model="fake-embed", timestamp="T0")

    meta = index.metadata
    assert {d.source_path for d in meta.documents} == {"a.txt", "b.txt"}
    assert meta.embedding_model == "fake-embed"
    assert meta.vector_dim == FakeEmbedder().dimension
    assert meta.created_at == "T0" and meta.updated_at == "T0"
    assert "AL-204" in index.code_index.values


def test_update_skips_unchanged_and_refreshes_changed() -> None:
    fs = _corpus()
    pipeline = _pipeline(fs, FakeEmbedder())
    index = pipeline.ingest(["a.txt", "b.txt"], embedding_model="fake", timestamp="T0")

    index, skipped = pipeline.update(
        index, ["a.txt", "b.txt"], embedding_model="fake", timestamp="T1"
    )
    assert set(skipped.skipped) == {"a.txt", "b.txt"}
    assert not skipped.added and not skipped.refreshed

    # Change a.txt; its identifiers should update on refresh.
    fs.add_text("a.txt", "ALARM REFERENCE\n\nAlarm 999 means overheat.\n")
    index, changed = pipeline.update(index, ["a.txt"], embedding_model="fake", timestamp="T2")
    assert changed.refreshed == ("a.txt",)
    assert "AL-999" in index.code_index.values
    assert "AL-204" not in index.code_index.values
    assert index.metadata.updated_at == "T2" and index.metadata.created_at == "T0"


def test_update_adds_new_documents() -> None:
    fs = _corpus()
    pipeline = _pipeline(fs, FakeEmbedder())
    index = pipeline.ingest(["a.txt"], embedding_model="fake", timestamp="T0")

    fs.add_text("c.txt", "PARTS\n\nThe drive belt is PN 44-19A.\n")
    index, summary = pipeline.update(index, ["c.txt"], embedding_model="fake", timestamp="T1")

    assert summary.added == ("c.txt",)
    assert {d.source_path for d in index.metadata.documents} == {"a.txt", "c.txt"}
    assert "PN-44-19A" in index.code_index.values


def test_remove_drops_document_and_its_chunks() -> None:
    fs = _corpus()
    pipeline = _pipeline(fs, FakeEmbedder())
    index = pipeline.ingest(["a.txt", "b.txt"], embedding_model="fake", timestamp="T0")
    before = len(index.chunks)

    index, summary = pipeline.remove(index, ["b.txt"], timestamp="T1")
    assert summary.removed == ("b.txt",)
    assert all(c.source_path != "b.txt" for c in index.chunks.values())
    assert {d.source_path for d in index.metadata.documents} == {"a.txt"}
    assert len(index.chunks) < before
    # The vector store dropped the removed chunks too.
    assert set(index.store.vectors()) == set(index.chunks)


def test_remove_matches_by_file_name() -> None:
    fs = _corpus()
    pipeline = _pipeline(fs, FakeEmbedder())
    index = pipeline.ingest(["a.txt", "b.txt"], embedding_model="fake", timestamp="T0")

    index, summary = pipeline.remove(index, ["a.txt"], timestamp="T1")
    assert summary.removed == ("a.txt",)


def test_update_rejects_a_different_embedding_model() -> None:
    fs = _corpus()
    pipeline = _pipeline(fs, FakeEmbedder())
    index = pipeline.ingest(["a.txt"], embedding_model="model-one", timestamp="T0")

    with pytest.raises(CogwrightError):
        pipeline.update(index, ["b.txt"], embedding_model="model-two", timestamp="T1")


def test_metadata_survives_serialization_and_is_backward_compatible() -> None:
    fs = _corpus()
    pipeline = _pipeline(fs, FakeEmbedder())
    index = pipeline.ingest(["a.txt", "b.txt"], embedding_model="fake", timestamp="T0")

    restored = Index.from_dict(index.to_dict())
    assert restored.metadata.embedding_model == "fake"
    assert {d.source_path for d in restored.metadata.documents} == {"a.txt", "b.txt"}

    # A version-1 index without a metadata block still loads, with empty metadata.
    legacy = index.to_dict()
    legacy["version"] = 1
    legacy.pop("metadata")
    loaded = Index.from_dict(legacy)
    assert loaded.metadata.embedding_model == ""
    assert loaded.chunks.keys() == index.chunks.keys()
