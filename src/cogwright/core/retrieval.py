# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Hybrid retrieval: semantic search merged with exact identifier lookup.

Two signals are combined. Semantic top-k finds passages that are about the
question. Exact code lookup finds the passage that documents a specific alarm,
stop, fault, error, or part identifier mentioned in the query. Exact matches are
precise, so they are ranked above any purely semantic hit; a chunk that wins on
both signals is marked as a hybrid match and scored highest.

The function is deterministic given fixed vectors, which makes ranking testable
with injected embeddings and no live model.
"""

from __future__ import annotations

from .code_index import CodeIndexer
from .config import RetrievalConfig
from .index import Index
from .models import CodeRef, ScoredChunk, Vector


def retrieve(
    index: Index,
    query_vector: Vector,
    query_codes: tuple[CodeRef, ...],
    config: RetrievalConfig,
) -> list[ScoredChunk]:
    """Return ranked chunks for a query, blending semantic and code signals."""

    # Semantic candidates from the vector store.
    semantic: dict[str, float] = {}
    for chunk_id, score in index.store.search(query_vector, config.top_k):
        if score >= config.min_score:
            semantic[chunk_id] = score

    # Exact identifier matches. These are always included regardless of the
    # semantic threshold because an exact code hit is a precise answer.
    code_hits = index.code_index.lookup(query_codes)

    scored: dict[str, ScoredChunk] = {}

    for chunk_id, score in semantic.items():
        chunk = index.chunks.get(chunk_id)
        if chunk is None:
            continue
        scored[chunk_id] = ScoredChunk(chunk=chunk, score=score, match_type="semantic")

    for chunk_id in code_hits:
        chunk = index.chunks.get(chunk_id)
        if chunk is None:
            continue
        existing = scored.get(chunk_id)
        semantic_score = existing.score if existing is not None else 0.0
        match_type = "hybrid" if existing is not None else "code"
        scored[chunk_id] = ScoredChunk(
            chunk=chunk,
            score=config.code_base_score + semantic_score,
            match_type=match_type,
        )

    ranked = sorted(
        scored.values(),
        key=lambda sc: (-sc.score, sc.chunk.chunk_id),
    )
    return ranked


def extract_query_codes(
    query: str,
    indexer: CodeIndexer,
) -> tuple[CodeRef, ...]:
    """Detect identifiers in a raw query string for exact lookup."""

    return indexer.extract(query)
