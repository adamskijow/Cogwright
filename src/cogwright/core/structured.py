# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Optional structured-output mode: parse a JSON answer from the model.

A capable model can return the answer as a JSON object, which gives reliable
numbered steps and an explicit list of the passages it used, instead of relying
on prose formatting and citation scraping. Parsing is defensive: it tolerates
code fences and surrounding prose, and returns ``None`` when the model did not
produce usable JSON so the caller can fall back to the prose path. A small model
that fumbles the format therefore never breaks a query.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# Strips a leading ```json (or ```) fence and a trailing ``` fence.
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

# A number or bullet a model sometimes bakes into a step string; removed so the
# rendered numbering is not doubled ("1. 1. Stop the unit").
_LEADING_ENUMERATOR = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")


def _clean_step(step: str) -> str:
    return _LEADING_ENUMERATOR.sub("", step.strip()).strip()


@dataclass(frozen=True)
class StructuredAnswer:
    """A model reply parsed from JSON."""

    found: bool
    answer: str
    steps: tuple[str, ...]
    used_ids: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.answer and not self.steps


def parse_structured(text: str, valid_ids: Iterable[str]) -> StructuredAnswer | None:
    """Parse a JSON answer, keeping only passage ids that were actually offered.

    Returns ``None`` when no JSON object can be recovered.
    """

    obj = _extract_json_object(text)
    if obj is None:
        return None

    valid = set(valid_ids)
    answer = obj.get("answer")
    raw_steps = obj.get("steps")
    raw_used = obj.get("used")

    steps = (
        tuple(cleaned for s in raw_steps if (cleaned := _clean_step(str(s))))
        if isinstance(raw_steps, list)
        else ()
    )
    # Models often echo the ids with their brackets ("[abc123]"), so normalize
    # before checking them against the passages that were actually offered.
    used = (
        tuple(
            cleaned
            for u in raw_used
            if isinstance(u, str) and (cleaned := u.strip().strip("[]")) in valid
        )
        if isinstance(raw_used, list)
        else ()
    )
    return StructuredAnswer(
        found=bool(obj.get("found", True)),
        answer=answer.strip() if isinstance(answer, str) else "",
        steps=steps,
        used_ids=used,
    )


def render_answer_text(structured: StructuredAnswer) -> str:
    """Render the parsed answer back into readable text with numbered steps."""

    parts: list[str] = []
    if structured.answer:
        parts.append(structured.answer)
    parts.extend(f"{i}. {step}" for i, step in enumerate(structured.steps, start=1))
    return "\n".join(parts).strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = _FENCE.sub("", text).strip()
    for candidate in (stripped, _outermost_braces(stripped)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _outermost_braces(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return None
