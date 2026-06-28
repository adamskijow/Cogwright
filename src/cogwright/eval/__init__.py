# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""A small, deterministic evaluation harness for retrieval quality.

It measures the parts of the system that do not depend on the language model:
whether retrieval surfaces the right source page, whether a named identifier
resolves, and whether the not-found decision is correct. Running it does not call
the model, so a graded dataset gives a reproducible quality signal.
"""

from __future__ import annotations

from .dataset import EvalCase, parse_dataset
from .metrics import CaseOutcome, EvalReport, evaluate

__all__ = [
    "CaseOutcome",
    "EvalCase",
    "EvalReport",
    "evaluate",
    "parse_dataset",
]
