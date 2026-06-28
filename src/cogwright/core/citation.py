# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Mapping answers back to their source locations (pure).

The model is asked to cite passages by their bracketed chunk id. The
:class:`CitationMapper` reads those markers out of the answer text and resolves
each to a :class:`Citation` with the source path, page, and section. When the
model cites nothing recognizable, the mapper falls back to the retrieved chunks
so an answer is never left without provenance.
"""

from __future__ import annotations

import re
from typing import Sequence

from .models import Chunk, Citation, ScoredChunk

# Matches the bracketed ids the prompt instructs the model to emit, e.g. [1a2b3c4d].
_CITATION_PATTERN = re.compile(r"\[([0-9a-f]{6,32})\]")


class CitationMapper:
    """Resolves chunk ids to citations against a known chunk set."""

    def __init__(self, chunks: dict[str, Chunk]) -> None:
        self._chunks = chunks

    def citation_for(self, chunk_id: str) -> Citation | None:
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return None
        return Citation(
            chunk_id=chunk.chunk_id,
            source_path=chunk.source_path,
            page=chunk.page,
            section=chunk.section,
        )

    def extract_cited_ids(self, answer_text: str) -> list[str]:
        """Return the in-corpus chunk ids cited in ``answer_text``, in order."""

        result: list[str] = []
        seen: set[str] = set()
        for match in _CITATION_PATTERN.finditer(answer_text):
            chunk_id = match.group(1)
            if chunk_id in self._chunks and chunk_id not in seen:
                seen.add(chunk_id)
                result.append(chunk_id)
        return result

    def map_answer(
        self,
        answer_text: str,
        retrieved: Sequence[ScoredChunk],
    ) -> tuple[Citation, ...]:
        """Resolve the citations for an answer.

        Prefers the ids the model actually cited; if it cited none that we
        recognize, falls back to the retrieved passages so the answer still
        points at where it came from.
        """

        cited_ids = self.extract_cited_ids(answer_text)
        if cited_ids:
            citations = [self.citation_for(cid) for cid in cited_ids]
            return tuple(c for c in citations if c is not None)
        fallback: list[Citation] = []
        seen: set[str] = set()
        for scored in retrieved:
            if scored.chunk.chunk_id in seen:
                continue
            seen.add(scored.chunk.chunk_id)
            fallback.append(
                Citation(
                    chunk_id=scored.chunk.chunk_id,
                    source_path=scored.chunk.source_path,
                    page=scored.chunk.page,
                    section=scored.chunk.section,
                )
            )
        return tuple(fallback)
