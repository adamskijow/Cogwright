# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for identifier detection and the exact-lookup code index."""

from __future__ import annotations

import pytest

from cogwright.core.code_index import CodeIndex, CodeIndexer
from cogwright.core.config import DEFAULT_CODE_PATTERNS
from cogwright.core.models import CodeRef

from .builders import chunk


@pytest.fixture
def indexer() -> CodeIndexer:
    return CodeIndexer(DEFAULT_CODE_PATTERNS)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Clear alarm 204 and restart.", "AL-204"),
        ("See AL-204 in the table.", "AL-204"),
        ("Code AL204 appears on the panel.", "AL-204"),
        ("Press reset for STOP CODE 12.", "SC-12"),
        ("Resolve SC-12 before continuing.", "SC-12"),
        ("Fault 09 means low pressure.", "F-09"),
        ("Error 30 indicates a sensor issue.", "E-30"),
        ("Order part PN 44-19A for the belt.", "PN-44-19A"),
        ("Replace P/N 44-19A as needed.", "PN-44-19A"),
        ("Part Number 7788-01 is the seal kit.", "PN-7788-01"),
        ("Diagnostic DTC P0420 was logged.", "DTC-P0420"),
        ("See DTC-1234 in the log.", "DTC-1234"),
        ("Warning 18 means the filter is dirty.", "W-18"),
        ("Clear W-18 after service.", "W-18"),
    ],
)
def test_detects_and_normalizes(
    indexer: CodeIndexer, text: str, expected: str
) -> None:
    values = [code.value for code in indexer.extract(text)]
    assert expected in values


def test_does_not_match_bare_words_or_numbers(indexer: CodeIndexer) -> None:
    # "shelf 023" must not be misread as fault F-023, and a bare number on its
    # own carries no identifier prefix to anchor a match.
    assert indexer.extract("Place it on shelf 023 in the rack.") == ()
    assert indexer.extract("The clearance is 12 millimeters.") == ()


def test_raw_text_is_preserved(indexer: CodeIndexer) -> None:
    (code,) = indexer.extract("Clear Alarm 204 now.")
    assert code.kind == "alarm"
    assert code.value == "AL-204"
    assert code.raw == "Alarm 204"


def test_extract_is_deduplicated_and_ordered(indexer: CodeIndexer) -> None:
    codes = indexer.extract("AL-204 then SC-12 then AL-204 again.")
    assert [c.value for c in codes] == ["AL-204", "SC-12"]


def test_code_index_lookup_resolves_to_chunks() -> None:
    alarm = CodeRef(kind="alarm", value="AL-204", raw="AL-204")
    part = CodeRef(kind="part", value="PN-44-19A", raw="PN 44-19A")
    a = chunk("aaa", codes=(alarm,))
    b = chunk("bbb", codes=(part,))
    c = chunk("ccc", codes=(alarm,))
    code_index = CodeIndex.from_chunks([a, b, c])

    assert code_index.lookup([alarm]) == ["aaa", "ccc"]
    assert code_index.lookup([part]) == ["bbb"]
    assert code_index.lookup([CodeRef("alarm", "AL-999", "AL-999")]) == []
    assert "AL-204" in code_index.values
