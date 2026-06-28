# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Detection and indexing of alarm, stop, fault, error, and part identifiers.

A dedicated pass treats these identifiers as first-class lookup keys. A query
that is a bare code (for example ``AL-204`` or ``alarm 204``) then resolves to
the exact passage that documents it, instead of relying on fuzzy semantic match.

The logic is pure and driven entirely by the :class:`CodePattern` rules in the
config, so the identifier schemes a deployment cares about are data, not code.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from .config import CodePattern
from .models import Chunk, CodeRef

# Guard so an identifier prefix is not picked up in the middle of a word or
# number. Without it, "shelf 023" could be misread as fault "F-023".
_LEADING_GUARD = r"(?<![A-Za-z0-9])"


class CodeIndexer:
    """Compiles code patterns and extracts normalized identifiers from text."""

    def __init__(self, patterns: Sequence[CodePattern]) -> None:
        self._patterns = tuple(patterns)
        self._compiled: list[tuple[CodePattern, re.Pattern[str]]] = [
            (p, re.compile(_LEADING_GUARD + p.regex, re.IGNORECASE)) for p in patterns
        ]

    @property
    def patterns(self) -> tuple[CodePattern, ...]:
        return self._patterns

    def extract(self, text: str) -> tuple[CodeRef, ...]:
        """Return the identifiers found in ``text``, de-duplicated, in order.

        When two patterns match at the same span, the earlier pattern in the
        configuration wins, so more specific rules can be ordered first.
        """

        # span-start -> (pattern order, CodeRef); keep the first pattern to claim
        # a given starting position so overlapping rules do not double count.
        claimed: dict[int, tuple[int, CodeRef]] = {}
        for order, (pattern, compiled) in enumerate(self._compiled):
            for match in compiled.finditer(text):
                identifier = match.group("id")
                if identifier is None:
                    continue
                value = self._normalize(pattern, identifier)
                ref = CodeRef(kind=pattern.name, value=value, raw=match.group(0).strip())
                start = match.start()
                existing = claimed.get(start)
                if existing is None or order < existing[0]:
                    claimed[start] = (order, ref)

        seen: set[str] = set()
        result: list[CodeRef] = []
        for _, (_, ref) in sorted(claimed.items()):
            if ref.value in seen:
                continue
            seen.add(ref.value)
            result.append(ref)
        return tuple(result)

    @staticmethod
    def _normalize(pattern: CodePattern, identifier: str) -> str:
        return f"{pattern.canonical_prefix}-{identifier.upper()}"


class CodeIndex:
    """Maps a normalized identifier to the chunks that mention it."""

    def __init__(self) -> None:
        self._by_value: dict[str, list[str]] = {}

    def add_chunk(self, chunk: Chunk) -> None:
        for code in chunk.codes:
            bucket = self._by_value.setdefault(code.value, [])
            if chunk.chunk_id not in bucket:
                bucket.append(chunk.chunk_id)

    def lookup(self, codes: Iterable[CodeRef]) -> list[str]:
        """Return chunk ids that mention any of ``codes``, preserving order."""

        result: list[str] = []
        seen: set[str] = set()
        for code in codes:
            for chunk_id in self._by_value.get(code.value, ()):
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    result.append(chunk_id)
        return result

    @property
    def values(self) -> frozenset[str]:
        return frozenset(self._by_value)

    @classmethod
    def from_chunks(cls, chunks: Iterable[Chunk]) -> "CodeIndex":
        index = cls()
        for chunk in chunks:
            index.add_chunk(chunk)
        return index
