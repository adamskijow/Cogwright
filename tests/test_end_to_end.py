# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""End-to-end: ingest the sample manual from disk and ask against it.

The document is parsed off the real filesystem with the real text parser; only
the model and embedder are faked, so the full retrieval, grounding, citation, and
not-found behavior is exercised without any network call.
"""

from __future__ import annotations

import json
from pathlib import Path

from cogwright.adapters.filesystem import RealFileSystem
from cogwright.adapters.text_parser import TextDocumentParser
from cogwright.core.config import Config
from cogwright.core.engine import IngestionPipeline, QueryEngine
from cogwright.core.index import Index
from cogwright.core.prompt import NOT_FOUND_MESSAGE

from .fakes import FakeEmbedder, FakeLLMClient

FIXTURE = Path(__file__).parent / "fixtures" / "sample_manual" / "series7_conveyor_manual.txt"


def _ingest(embedder: FakeEmbedder, config: Config) -> Index:
    pipeline = IngestionPipeline(
        RealFileSystem(), [TextDocumentParser()], embedder, config
    )
    return pipeline.ingest([str(FIXTURE)])


def test_alarm_query_yields_cited_step_by_step_answer() -> None:
    embedder = FakeEmbedder()
    config = Config()
    index = _ingest(embedder, config)

    llm = FakeLLMClient()
    answer = QueryEngine(embedder, llm, config).ask(index, "How do I clear alarm 204?")

    assert answer.found is True
    # A numbered, step-by-step procedure.
    assert "1." in answer.text and "2." in answer.text
    # The bare alarm code resolved to the exact passage that documents it.
    assert "AL-204" in {c.value for c in answer.referenced_codes}
    # Cited back to the page that carries the alarm reference.
    assert any(c.page == 3 for c in answer.citations)
    assert any("ALARM" in (c.section or "") for c in answer.citations)


def test_part_number_resolves_to_the_parts_table() -> None:
    embedder = FakeEmbedder()
    config = Config()
    index = _ingest(embedder, config)

    answer = QueryEngine(embedder, FakeLLMClient(), config).ask(
        index, "What is part PN 44-19A used for?"
    )

    assert answer.found is True
    assert "PN-44-19A" in {c.value for c in answer.referenced_codes}
    # The replacement-parts table lives on page 4.
    assert any(c.page == 4 for c in answer.citations)


def test_unanswerable_question_returns_not_found_without_calling_model() -> None:
    embedder = FakeEmbedder()
    config = Config()
    index = _ingest(embedder, config)

    llm = FakeLLMClient()
    answer = QueryEngine(embedder, llm, config).ask(
        index, "Bluetooth pairing instructions"
    )

    assert answer.found is False
    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.citations == ()
    # No invented procedure: the model is never reached.
    assert llm.calls == []


def test_persisted_index_round_trips_through_disk(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    config = Config()
    index = _ingest(embedder, config)

    fs = RealFileSystem()
    index_path = str(tmp_path / "index.json")
    fs.write_text(index_path, json.dumps(index.to_dict()))

    reloaded = Index.from_dict(json.loads(fs.read_text(index_path)))
    answer = QueryEngine(embedder, FakeLLMClient(), config).ask(
        reloaded, "How do I clear alarm 204?"
    )
    assert answer.found is True
    assert any(c.page == 3 for c in answer.citations)
