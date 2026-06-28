# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""A pure, in-memory vector store: the reference implementation of the seam.

It scores candidates with cosine similarity in plain Python so the core carries
no numerical or vector-database dependency. The :class:`VectorStore` protocol
lets a deployment swap in an approximate-nearest-neighbor index later without
changing any retrieval code.
"""

from __future__ import annotations

import math
from typing import Mapping

from .models import Vector


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity of two equal-length vectors, in ``[-1.0, 1.0]``.

    Returns ``0.0`` when either vector has zero magnitude or the lengths differ,
    so callers never divide by zero or crash on a malformed embedding.
    """

    if len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class InMemoryVectorStore:
    """Keeps every chunk vector in a dict and scans them on each query.

    This is exact and deterministic, which is what the M1 corpus sizes call for
    and what makes retrieval straightforward to test.
    """

    def __init__(self) -> None:
        self._vectors: dict[str, Vector] = {}

    def add(self, chunk_id: str, vector: Vector) -> None:
        self._vectors[chunk_id] = list(vector)

    def search(self, vector: Vector, k: int) -> list[tuple[str, float]]:
        scored = [
            (chunk_id, cosine_similarity(vector, stored))
            for chunk_id, stored in self._vectors.items()
        ]
        # Sort by score descending, then by chunk id so ties are deterministic.
        scored.sort(key=lambda item: (-item[1], item[0]))
        if k >= 0:
            return scored[:k]
        return scored

    def vectors(self) -> Mapping[str, Vector]:
        return dict(self._vectors)

    def load(self, data: Mapping[str, Vector]) -> None:
        self._vectors = {chunk_id: list(vector) for chunk_id, vector in data.items()}

    def __len__(self) -> int:
        return len(self._vectors)
