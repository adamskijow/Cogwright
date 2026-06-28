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
from collections.abc import Sequence

from .models import Chunk, Citation, ScoredChunk

# Chunk ids are bare hexadecimal tokens. Scanning for them directly, rather than
# requiring one exact bracket format, tolerates the varied ways a real model
# cites: [id], [id1, id2], (see id), or inline. False matches are ruled out by
# checking each token against the known chunk ids.
_ID_TOKEN = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{6,32}(?![0-9a-fA-F])")

# A bracketed run of one or more ids, removed when cleaning an answer for display.
_CITATION_GROUP = re.compile(r"\s*\[[0-9a-f][0-9a-f,;\s]*\]")

# How many top-ranked passages to cite when the model itself cited none.
_MAX_FALLBACK_CITATIONS = 3


def strip_citation_markers(text: str) -> str:
    """Remove bracketed citation groups for clean display, leaving prose intact."""

    cleaned = _CITATION_GROUP.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +\n", "\n", cleaned)
    return cleaned.strip()


def clean_answer_text(text: str, drop_line: str) -> str:
    """Prepare an answered reply for display.

    Strips citation markers and removes any standalone line equal to
    ``drop_line``. Small models sometimes append the not-found sentence even
    after answering; passing it here keeps it out of a grounded reply.
    """

    target = drop_line.strip()
    lines = [
        line
        for line in strip_citation_markers(text).splitlines()
        if line.strip() != target
    ]
    return "\n".join(lines).strip()


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
        for match in _ID_TOKEN.finditer(answer_text):
            chunk_id = match.group(0)
            if chunk_id in self._chunks and chunk_id not in seen:
                seen.add(chunk_id)
                result.append(chunk_id)
        return result

    def map_answer(
        self,
        answer_text: str,
        retrieved: Sequence[ScoredChunk],
        max_fallback: int = _MAX_FALLBACK_CITATIONS,
    ) -> tuple[Citation, ...]:
        """Resolve the citations for an answer.

        Prefers the ids the model actually cited. If it cited none that we
        recognize, which smaller models often do, falls back to the highest
        ranked retrieved passages, capped so a weakly grounded answer is not
        decorated with every passage that was in context.
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
            if len(fallback) >= max_fallback:
                break
        return tuple(fallback)
