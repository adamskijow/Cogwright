# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""The graded dataset format for evaluation.

A dataset is a list of cases, each pairing a question with what a correct
retrieval should surface: the source pages that contain the answer, the
identifiers that should resolve, and whether the question is answerable at all
from the corpus. Parsing is pure and validates shape so a malformed file fails
loudly rather than silently scoring wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DATASET_VERSION = 1


@dataclass(frozen=True)
class EvalCase:
    """One graded question and its expected retrieval outcome."""

    question: str
    expected_pages: tuple[int, ...] = ()
    expected_codes: tuple[str, ...] = ()
    should_find: bool = True


def parse_dataset(data: dict[str, Any]) -> list[EvalCase]:
    """Parse a dataset dict (already loaded from JSON) into cases."""

    version = data.get("version")
    if version != DATASET_VERSION:
        raise ValueError(
            f"unsupported eval dataset version {version!r}; expected {DATASET_VERSION}"
        )
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("eval dataset must contain a 'cases' list")
    return [_parse_case(i, raw) for i, raw in enumerate(raw_cases)]


def _parse_case(index: int, raw: object) -> EvalCase:
    if not isinstance(raw, dict):
        raise ValueError(f"case {index} must be an object")
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"case {index} must have a non-empty 'question'")
    pages = tuple(int(p) for p in raw.get("expected_pages", ()))
    codes = tuple(str(c) for c in raw.get("expected_codes", ()))
    should_find = bool(raw.get("should_find", True))
    return EvalCase(
        question=question,
        expected_pages=pages,
        expected_codes=codes,
        should_find=should_find,
    )
