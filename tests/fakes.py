# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Deterministic test doubles for every protocol seam.

No live model, endpoint, or disk is touched in the unit suite. The embedder is a
stable bag-of-words projection so semantic similarity is meaningful yet exactly
reproducible, and the model is a grounded fake that reads the prompt context and
echoes the relevant steps with a real citation, so end-to-end behavior is tested
without any network call.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence

from cogwright.core.models import Document, Message, Vector

_TOKEN = re.compile(r"[a-z0-9]+")
_STEP_LINE = re.compile(r"^\s*(?:\d+[.)]\s|step\s+\d+)", re.IGNORECASE)
_PASSAGE_HEADER = re.compile(r"^\[([0-9a-f]{6,32})\]\s*\(")

# Dropping function words keeps similarity driven by content, so a question with
# no content overlap with the corpus is genuinely orthogonal, not falsely linked
# by shared words like "the". This mirrors how a real embedding model behaves and
# makes the not-found path deterministic in tests.
_STOPWORDS = frozenset(
    """a an and are as at be by do does for from how i in into is it its me my no
    not of on or please should that the then than this to we what when where which
    why will with you your every before after if can must""".split()
)


def _tokenize(text: str) -> list[str]:
    return [
        tok
        for tok in _TOKEN.findall(text.lower())
        if len(tok) >= 2 and tok not in _STOPWORDS
    ]


class FakeEmbedder:
    """A deterministic bag-of-words embedder with one slot per distinct token.

    Each distinct content token is assigned its own dimension the first time it is
    seen, so two different tokens never collide. Tokens shared between a query and
    a chunk land in the same slot and drive similarity; a token that appears only
    in the query gets a fresh slot and contributes nothing, so an unrelated
    question scores zero against every chunk.
    """

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []
        self._slots: dict[str, int] = {}

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> Vector:
        vector = [0.0] * self.dimension
        for token in _tokenize(text):
            vector[self._slot(token)] += 1.0
        return vector

    def _slot(self, token: str) -> int:
        slot = self._slots.get(token)
        if slot is None:
            # Sequential assignment guarantees distinct slots for distinct tokens
            # as long as the vocabulary stays under the dimension, which holds for
            # the small corpora used in tests.
            slot = len(self._slots) % self.dimension
            self._slots[token] = slot
        return slot


class FakeFileSystem:
    """An in-memory filesystem keyed by path string."""

    def __init__(self, files: Mapping[str, bytes] | None = None) -> None:
        self._files: dict[str, bytes] = dict(files or {})

    def add_text(self, path: str, text: str) -> None:
        self._files[path] = text.encode("utf-8")

    def read_bytes(self, path: str) -> bytes:
        return self._files[path]

    def read_text(self, path: str) -> str:
        return self._files[path].decode("utf-8")

    def write_text(self, path: str, data: str) -> None:
        self._files[path] = data.encode("utf-8")

    def exists(self, path: str) -> bool:
        return path in self._files or any(
            key.startswith(path.rstrip("/") + "/") for key in self._files
        )

    def list_files(self, path: str, extensions: Sequence[str]) -> list[str]:
        prefix = path.rstrip("/") + "/"
        matches = [
            key
            for key in self._files
            if key == path or key.startswith(prefix)
        ]
        if extensions:
            lowered = tuple(ext.lower() for ext in extensions)
            matches = [m for m in matches if m.lower().endswith(lowered)]
        return sorted(matches)


class FakeDocumentParser:
    """Returns preconfigured documents for known paths."""

    def __init__(
        self,
        documents: Mapping[str, Document],
        extensions: Sequence[str] = (".txt",),
    ) -> None:
        self._documents = dict(documents)
        self._extensions = tuple(extensions)

    def supports(self, path: str) -> bool:
        return path in self._documents or path.lower().endswith(self._extensions)

    def parse(self, path: str, data: bytes) -> Document:
        return self._documents[path]


class FakeLLMClient:
    """A grounded fake model.

    It reads the rendered context passages from the prompt, finds the first
    passage that contains numbered steps, and returns those steps with a real
    citation to that passage. With no steps present it answers from the first
    passage's leading line. It records every call so tests can assert the
    not-found path never reaches the model.
    """

    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message]) -> str:
        self.calls.append(list(messages))
        return self._answer(messages)

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        # Yield the full answer in two pieces so streaming consumers are exercised.
        answer = self.complete(messages)
        midpoint = len(answer) // 2
        yield answer[:midpoint]
        yield answer[midpoint:]

    def available(self) -> bool:
        return self._available

    def _answer(self, messages: Sequence[Message]) -> str:
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        passages = _parse_passages(user)
        if not passages:
            return "I could not find an answer to that in the provided documents."
        for chunk_id, body in passages:
            steps = [line for line in body if _STEP_LINE.match(line)]
            if steps:
                joined = "\n".join(steps)
                return f"Follow these steps:\n{joined}\n[{chunk_id}]"
        chunk_id, body = passages[0]
        lead = next((line for line in body if line.strip()), "")
        return f"{lead} [{chunk_id}]"


def _parse_passages(user_message: str) -> list[tuple[str, list[str]]]:
    """Recover (chunk_id, body_lines) pairs from a rendered user prompt."""

    passages: list[tuple[str, list[str]]] = []
    current_id: str | None = None
    body: list[str] = []
    for line in user_message.splitlines():
        header = _PASSAGE_HEADER.match(line)
        if header:
            if current_id is not None:
                passages.append((current_id, body))
            current_id = header.group(1)
            body = []
            continue
        if current_id is not None:
            if line.startswith("Identifiers:") or line.strip() == "(table)":
                continue
            body.append(line)
    if current_id is not None:
        passages.append((current_id, body))
    return passages
