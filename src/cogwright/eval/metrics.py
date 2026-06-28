# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Running graded cases and scoring retrieval quality (pure).

Scoring uses only the retrieval decision, never a model call, so the numbers are
reproducible. Three signals are measured: did retrieval surface a correct source
page, did the named identifiers resolve, and was the answerable / not-answerable
decision right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..core.engine import Preparation
from .dataset import EvalCase

# A function that runs retrieval for a question and returns the Preparation,
# typically ``lambda q: engine.prepare(index, q)``.
Preparer = Callable[[str], Preparation]


@dataclass(frozen=True)
class CaseOutcome:
    """The graded result for a single case."""

    case: EvalCase
    found: bool
    found_correct: bool
    retrieved_pages: tuple[int, ...]
    page_hit: bool | None  # None when the case specifies no expected pages
    codes_resolved: bool | None  # None when the case specifies no expected codes


@dataclass(frozen=True)
class Rate:
    """A hits-over-applicable ratio, carrying its counts for transparency."""

    hits: int
    total: int

    @property
    def value(self) -> float:
        # A rate over zero applicable cases is vacuously satisfied.
        return self.hits / self.total if self.total else 1.0


@dataclass(frozen=True)
class EvalReport:
    """Per-case outcomes and the aggregate rates over a dataset."""

    outcomes: tuple[CaseOutcome, ...]

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def found_accuracy(self) -> Rate:
        values = [o.found_correct for o in self.outcomes]
        return Rate(sum(values), len(values))

    @property
    def page_hit_rate(self) -> Rate:
        values = [o.page_hit for o in self.outcomes if o.page_hit is not None]
        return Rate(sum(values), len(values))

    @property
    def code_resolution_rate(self) -> Rate:
        values = [o.codes_resolved for o in self.outcomes if o.codes_resolved is not None]
        return Rate(sum(values), len(values))

    @property
    def not_found_accuracy(self) -> Rate:
        values = [not o.found for o in self.outcomes if not o.case.should_find]
        return Rate(sum(values), len(values))

    def summary(self) -> str:
        lines = [
            f"cases:               {self.total}",
            _line("found accuracy", self.found_accuracy),
            _line("page hit rate", self.page_hit_rate),
            _line("code resolution", self.code_resolution_rate),
            _line("not-found accuracy", self.not_found_accuracy),
        ]
        return "\n".join(lines)


def _line(label: str, rate: Rate) -> str:
    return f"{label + ':':20} {rate.value:6.1%}  ({rate.hits}/{rate.total})"


def evaluate(cases: Sequence[EvalCase], prepare: Preparer) -> EvalReport:
    """Score every case and return the aggregate report."""

    return EvalReport(tuple(_score(case, prepare) for case in cases))


def _score(case: EvalCase, prepare: Preparer) -> CaseOutcome:
    prep = prepare(case.question)
    retrieved_pages = tuple(sc.chunk.page for sc in prep.retrieved)
    page_set = set(retrieved_pages)

    page_hit: bool | None = None
    if case.expected_pages:
        page_hit = any(page in page_set for page in case.expected_pages)

    codes_resolved: bool | None = None
    if case.expected_codes:
        available = {code.value for code in prep.resolved_codes}
        for sc in prep.retrieved:
            available.update(code.value for code in sc.chunk.codes)
        codes_resolved = all(code in available for code in case.expected_codes)

    return CaseOutcome(
        case=case,
        found=prep.found,
        found_correct=prep.found == case.should_find,
        retrieved_pages=retrieved_pages,
        page_hit=page_hit,
        codes_resolved=codes_resolved,
    )
