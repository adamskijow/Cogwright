# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for hybrid retrieval ranking using injected vectors."""

from __future__ import annotations

from cogwright.core.config import RetrievalConfig
from cogwright.core.models import CodeRef
from cogwright.core.retrieval import retrieve

from .builders import chunk, index_with

CONFIG = RetrievalConfig(top_k=6, min_score=0.15, code_base_score=1.0)
ALARM = CodeRef(kind="alarm", value="AL-5", raw="AL-5")


def test_semantic_results_rank_by_cosine_and_respect_threshold() -> None:
    a = chunk("a", text="aligned")
    b = chunk("b", text="weak")
    c = chunk("c", text="mixed")
    index = index_with({a: [1.0, 0.0, 0.0], b: [0.0, 1.0, 0.0], c: [1.0, 1.0, 0.0]})

    results = retrieve(index, [2.0, 1.0, 0.0], (), CONFIG)
    order = [r.chunk.chunk_id for r in results]

    # cosine: c (0.95) > a (0.89) > b (0.45); all clear the 0.15 threshold.
    assert order == ["c", "a", "b"]
    assert all(r.match_type == "semantic" for r in results)


def test_below_threshold_semantic_is_dropped() -> None:
    a = chunk("a", text="aligned")
    orthogonal = chunk("z", text="unrelated")
    index = index_with({a: [1.0, 0.0, 0.0], orthogonal: [0.0, 0.0, 1.0]})

    results = retrieve(index, [1.0, 0.0, 0.0], (), CONFIG)

    assert [r.chunk.chunk_id for r in results] == ["a"]


def test_code_hit_outranks_pure_semantic() -> None:
    coded = chunk("coded", text="alarm passage", codes=(ALARM,))
    semantic = chunk("sem", text="related")
    index = index_with({coded: [0.0, 1.0, 0.0], semantic: [1.0, 1.0, 0.0]})

    # Query aligns better with `semantic` semantically, but names the code in `coded`.
    results = retrieve(index, [1.0, 1.0, 0.0], (ALARM,), CONFIG)

    assert results[0].chunk.chunk_id == "coded"
    assert results[0].score > 1.0
    # `coded` also has a (weak) semantic score, so it is a hybrid match.
    assert results[0].match_type == "hybrid"


def test_code_only_match_is_included_even_below_threshold() -> None:
    coded = chunk("coded", text="alarm passage", codes=(ALARM,))
    other = chunk("other", text="something")
    # `coded` is orthogonal to the query, so it would never survive semantically.
    index = index_with({coded: [0.0, 0.0, 1.0], other: [1.0, 0.0, 0.0]})

    results = retrieve(index, [1.0, 0.0, 0.0], (ALARM,), CONFIG)
    by_id = {r.chunk.chunk_id: r for r in results}

    assert "coded" in by_id
    assert by_id["coded"].match_type == "code"
    assert by_id["coded"].score == CONFIG.code_base_score
    # And it ranks above the only weakly-related semantic hit.
    assert results[0].chunk.chunk_id == "coded"


def test_not_found_when_nothing_relevant() -> None:
    a = chunk("a", text="one")
    b = chunk("b", text="two")
    index = index_with({a: [1.0, 0.0, 0.0], b: [0.0, 1.0, 0.0]})

    # Orthogonal query, no codes: nothing clears the threshold and nothing is
    # resolved by code, so retrieval is empty and the engine will report not-found.
    results = retrieve(index, [0.0, 0.0, 1.0], (), CONFIG)

    assert results == []
