# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for mapping answers back to source citations."""

from __future__ import annotations

from cogwright.core.citation import CitationMapper, strip_citation_markers
from cogwright.core.models import ScoredChunk

from .builders import chunk


def _mapper() -> CitationMapper:
    a = chunk("a1b2c3d4", page=2, section="STARTUP", source_path="m.txt")
    b = chunk("e5f6a7b8", page=3, section="ALARMS", source_path="m.txt")
    return CitationMapper({a.chunk_id: a, b.chunk_id: b})


def test_extract_cited_ids_filters_unknown_ids() -> None:
    mapper = _mapper()
    cited = mapper.extract_cited_ids("Do this [a1b2c3d4] and see [e5f6a7b8]; ignore [deadbeef].")
    assert cited == ["a1b2c3d4", "e5f6a7b8"]


def test_citation_for_maps_id_to_page_and_section() -> None:
    mapper = _mapper()
    citation = mapper.citation_for("e5f6a7b8")
    assert citation is not None
    assert citation.page == 3
    assert citation.section == "ALARMS"
    assert citation.source_path == "m.txt"


def test_citation_for_unknown_id_is_none() -> None:
    assert _mapper().citation_for("ffffffff") is None


def test_map_answer_prefers_model_citations() -> None:
    mapper = _mapper()
    retrieved = [
        ScoredChunk(chunk=chunk("a1b2c3d4", page=2), score=1.0, match_type="semantic"),
        ScoredChunk(chunk=chunk("e5f6a7b8", page=3), score=0.5, match_type="semantic"),
    ]
    citations = mapper.map_answer("Use step two [e5f6a7b8].", retrieved)
    assert [c.chunk_id for c in citations] == ["e5f6a7b8"]


def test_map_answer_falls_back_to_retrieved_when_no_markers() -> None:
    mapper = _mapper()
    retrieved = [
        ScoredChunk(chunk=chunk("a1b2c3d4", page=2), score=1.0, match_type="semantic"),
        ScoredChunk(chunk=chunk("e5f6a7b8", page=3), score=0.5, match_type="semantic"),
    ]
    citations = mapper.map_answer("An answer with no bracketed ids.", retrieved)
    assert [c.chunk_id for c in citations] == ["a1b2c3d4", "e5f6a7b8"]


def test_extract_handles_multi_id_and_inline_citations() -> None:
    mapper = _mapper()
    cited = mapper.extract_cited_ids(
        "See [a1b2c3d4, e5f6a7b8] and a1b2c3d4 again; ignore deadbeefdeadbeef."
    )
    assert cited == ["a1b2c3d4", "e5f6a7b8"]


def test_strip_citation_markers_removes_brackets_and_tidies_spacing() -> None:
    text = "Do step one [a1b2c3d4] then step two [a1b2c3d4, e5f6a7b8]."
    assert strip_citation_markers(text) == "Do step one then step two."
