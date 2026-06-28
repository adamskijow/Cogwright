# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for the evaluation harness and the shipped graded dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cogwright.adapters.text_parser import TextDocumentParser
from cogwright.core.config import Config
from cogwright.core.engine import IngestionPipeline, QueryEngine
from cogwright.core.index import Index
from cogwright.eval import evaluate, parse_dataset
from cogwright.eval.dataset import EvalCase
from cogwright.eval.metrics import Rate

from .fakes import FakeEmbedder, FakeFileSystem, FakeLLMClient

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_manual"
MANUAL = FIXTURE_DIR / "series7_conveyor_manual.txt"
DATASET = FIXTURE_DIR / "eval.json"


def _index(embedder: FakeEmbedder, config: Config) -> Index:
    fs = FakeFileSystem()
    fs.add_text(str(MANUAL), MANUAL.read_text(encoding="utf-8"))
    pipeline = IngestionPipeline(fs, [TextDocumentParser()], embedder, config)
    return pipeline.ingest([str(MANUAL)])


def test_rate_handles_empty_denominator() -> None:
    assert Rate(0, 0).value == 1.0
    assert Rate(3, 4).value == 0.75


def test_parse_dataset_rejects_bad_version() -> None:
    with pytest.raises(ValueError):
        parse_dataset({"version": 99, "cases": []})


def test_parse_dataset_requires_question() -> None:
    with pytest.raises(ValueError):
        parse_dataset({"version": 1, "cases": [{"expected_pages": [1]}]})


def test_shipped_dataset_scores_perfectly_on_the_fixture() -> None:
    embedder = FakeEmbedder()
    config = Config()
    index = _index(embedder, config)
    engine = QueryEngine(embedder, FakeLLMClient(), config)

    cases = parse_dataset(json.loads(DATASET.read_text(encoding="utf-8")))
    report = evaluate(cases, lambda q: engine.prepare(index, q))

    assert report.total == len(cases)
    assert report.found_accuracy.value == 1.0
    assert report.page_hit_rate.value == 1.0
    assert report.code_resolution_rate.value == 1.0
    assert report.not_found_accuracy.value == 1.0
    # The not-found rate is measured over the two unanswerable cases.
    assert report.not_found_accuracy.total == 2


def test_report_flags_a_missed_page() -> None:
    embedder = FakeEmbedder()
    config = Config()
    index = _index(embedder, config)
    engine = QueryEngine(embedder, FakeLLMClient(), config)

    # The alarm answer is on page 3, so asserting page 1 must score as a miss.
    wrong = [EvalCase(question="How do I clear alarm 204?", expected_pages=(1,))]
    report = evaluate(wrong, lambda q: engine.prepare(index, q))

    assert report.page_hit_rate.value == 0.0
    assert "page hit rate" in report.summary()
