# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for the structured JSON answer parser and the engine's JSON mode."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence

from cogwright.core.config import Config
from cogwright.core.engine import IngestionPipeline, QueryEngine
from cogwright.core.index import Index
from cogwright.core.models import BlockKind, Message
from cogwright.core.structured import parse_structured, render_answer_text

from .builders import block, document
from .fakes import FakeDocumentParser, FakeEmbedder, FakeFileSystem, FakeLLMClient

VALID = {"a1b2c3d4", "e5f6a7b8"}


def test_parses_clean_json() -> None:
    raw = '{"found": true, "answer": "", "steps": ["Stop it", "Refill it"], "used": ["a1b2c3d4"]}'
    parsed = parse_structured(raw, VALID)
    assert parsed is not None
    assert parsed.found is True
    assert parsed.steps == ("Stop it", "Refill it")
    assert parsed.used_ids == ("a1b2c3d4",)


def test_tolerates_code_fences_and_surrounding_prose() -> None:
    raw = (
        "Sure!\n```json\n"
        '{"found": true, "answer": "Do X", "steps": [], "used": []}'
        "\n```\nHope that helps."
    )
    parsed = parse_structured(raw, VALID)
    assert parsed is not None
    assert parsed.answer == "Do X"


def test_drops_unknown_used_ids() -> None:
    raw = '{"found": true, "steps": ["one"], "used": ["a1b2c3d4", "deadbeef"]}'
    parsed = parse_structured(raw, VALID)
    assert parsed is not None
    assert parsed.used_ids == ("a1b2c3d4",)


def test_normalizes_bracketed_used_ids() -> None:
    # Smaller models echo the ids with their brackets, as they appear in context.
    raw = '{"found": true, "steps": ["one"], "used": ["[a1b2c3d4]", "[e5f6a7b8]"]}'
    parsed = parse_structured(raw, VALID)
    assert parsed is not None
    assert parsed.used_ids == ("a1b2c3d4", "e5f6a7b8")


def test_malformed_json_returns_none() -> None:
    assert parse_structured("not json at all", VALID) is None
    assert parse_structured("{broken: ", VALID) is None


def test_render_numbers_steps() -> None:
    raw = '{"found": true, "answer": "Here is how:", "steps": ["First", "Second"]}'
    parsed = parse_structured(raw, VALID)
    assert parsed is not None
    assert render_answer_text(parsed) == "Here is how:\n1. First\n2. Second"


def test_steps_with_their_own_numbering_are_not_doubled() -> None:
    # A model that bakes "1." into each step must not produce "1. 1. ...".
    raw = '{"found": true, "steps": ["1. Stop it", "3. Clear it"], "used": []}'
    parsed = parse_structured(raw, VALID)
    assert parsed is not None
    assert parsed.steps == ("Stop it", "Clear it")
    assert render_answer_text(parsed) == "1. Stop it\n2. Clear it"


PATH = "corpus/manual.txt"


def _index(embedder: FakeEmbedder, config: Config) -> Index:
    doc = document(
        "manual",
        block(BlockKind.HEADING, "ALARM REFERENCE", page=3),
        block(BlockKind.PARAGRAPH, "Alarm 204 indicates low coolant pressure.", page=3),
        block(BlockKind.STEP, "1. Stop the unit and let it cool.", page=3),
        block(BlockKind.STEP, "2. Clear alarm 204 and restart.", page=3),
        path=PATH,
    )
    fs = FakeFileSystem()
    fs.add_text(PATH, "ignored; the fake parser supplies the document")
    pipeline = IngestionPipeline(fs, [FakeDocumentParser({PATH: doc})], embedder, config)
    return pipeline.ingest([PATH])


class _JsonLLM:
    """A model that returns a JSON answer citing the first retrieved passage."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message]) -> str:
        self.calls.append(list(messages))
        user = next(m.content for m in reversed(messages) if m.role == "user")
        # Recover a passage id from the rendered context to cite it precisely.
        match = re.search(r"\[([0-9a-f]{6,32})\]", user)
        assert match is not None
        chunk_id = match.group(1)
        return (
            '{"found": true, "answer": "", '
            '"steps": ["Stop the unit", "Clear the alarm"], '
            f'"used": ["{chunk_id}"]}}'
        )

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        yield self.complete(messages)

    def available(self) -> bool:
        return True


def test_engine_json_mode_builds_precise_citations() -> None:
    embedder = FakeEmbedder()
    config = Config()
    index = _index(embedder, config)
    engine = QueryEngine(embedder, _JsonLLM(), config, structured=True)

    answer = engine.ask(index, "How do I clear alarm 204?")

    assert answer.found is True
    assert "1. Stop the unit" in answer.text
    assert len(answer.citations) == 1  # exactly the cited passage, not all retrieved
    assert answer.citations[0].page == 3


class _FixedLLM:
    """Returns a fixed reply regardless of the prompt."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def complete(self, messages: Sequence[Message]) -> str:
        return self._reply

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        yield self._reply

    def available(self) -> bool:
        return True


def test_found_false_with_content_is_still_answered() -> None:
    embedder = FakeEmbedder()
    config = Config()
    index = _index(embedder, config)
    # A small model that sets found=false but still returns steps; content wins.
    reply = '{"found": false, "answer": "", "steps": ["Stop it", "Clear it"], "used": []}'
    engine = QueryEngine(embedder, _FixedLLM(reply), config, structured=True)

    answer = engine.ask(index, "How do I clear alarm 204?")
    assert answer.found is True
    assert "1. Stop it" in answer.text


def test_found_false_and_empty_is_not_found() -> None:
    embedder = FakeEmbedder()
    config = Config()
    index = _index(embedder, config)
    reply = '{"found": false, "answer": "", "steps": [], "used": []}'
    engine = QueryEngine(embedder, _FixedLLM(reply), config, structured=True)

    answer = engine.ask(index, "How do I clear alarm 204?")
    assert answer.found is False


def test_engine_json_mode_falls_back_to_prose_on_bad_json() -> None:
    embedder = FakeEmbedder()
    config = Config()
    index = _index(embedder, config)
    # The default fake returns prose steps, not JSON, so structured parsing fails.
    engine = QueryEngine(embedder, FakeLLMClient(), config, structured=True)

    answer = engine.ask(index, "How do I clear alarm 204?")

    assert answer.found is True
    assert "1." in answer.text
    assert answer.citations  # fell back to prose assembly, still cited
