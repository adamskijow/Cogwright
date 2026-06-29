# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""The web application logic, independent of the HTTP layer.

:class:`WebApp` wraps the same engine the CLI uses and exposes the operations the
browser needs: index info, a streaming answer, and corpus changes (add a path,
add uploaded bytes, remove a document). It takes its collaborators as injected
protocols, so it is tested with fakes and never needs a live endpoint. All
mutations are serialized with a lock, since the threaded server may handle a
query and an ingest at once.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from ..core.citation import clean_answer_text
from ..core.config import Config
from ..core.engine import IngestionPipeline, QueryEngine
from ..core.index import Index, IndexMetadata
from ..core.models import Answer
from ..core.prompt import NOT_FOUND_MESSAGE
from ..core.protocols import DocumentParser, Embedder, FileSystem, LLMClient
from ..core.vector_store import InMemoryVectorStore


class WebApp:
    """Backs the browser UI with the retrieval engine and corpus operations."""

    def __init__(
        self,
        config: Config,
        index_path: str,
        fs: FileSystem,
        parsers: Sequence[DocumentParser],
        embedder: Embedder,
        llm_factory: Callable[[str], LLMClient],
    ) -> None:
        self._config = config
        self._index_path = index_path
        self._fs = fs
        self._embedder = embedder
        self._embedding_model = config.endpoint.embedding_model
        # The chat client is built from a factory so the model can be switched at
        # runtime; the embedder is fixed because its model is baked into the index.
        self._llm_factory = llm_factory
        self._pipeline = IngestionPipeline(fs, parsers, embedder, config)
        self._llm = llm_factory(config.endpoint.llm_model)
        self._engine = QueryEngine(embedder, self._llm, config)
        self._lock = threading.Lock()
        self._index = self._load_index()

    def update_settings(
        self, llm_model: str | None = None, min_score: float | None = None
    ) -> dict[str, Any]:
        """Switch the chat model or relevance cutoff live, then return the info.

        The embedding model is deliberately not switchable here: its vectors are
        baked into the index, so changing it means re-ingesting the corpus.
        """

        with self._lock:
            endpoint = self._config.endpoint
            retrieval = self._config.retrieval
            if llm_model and llm_model != endpoint.llm_model:
                endpoint = replace(endpoint, llm_model=llm_model)
                self._llm = self._llm_factory(llm_model)
            if min_score is not None:
                retrieval = replace(retrieval, min_score=max(0.0, min(1.0, min_score)))
            self._config = replace(
                self._config, endpoint=endpoint, retrieval=retrieval
            )
            self._engine = QueryEngine(self._embedder, self._llm, self._config)
        return self.info()

    def info(self) -> dict[str, Any]:
        """A JSON-ready description of the index and the active settings."""

        meta = self._index.metadata
        return {
            "documents": [
                {
                    "source_path": d.source_path,
                    "title": d.title,
                    "chunk_count": d.chunk_count,
                }
                for d in sorted(meta.documents, key=lambda r: r.source_path)
            ],
            "document_count": len(meta.documents),
            "chunk_count": len(self._index.chunks),
            "embedding_model": meta.embedding_model,
            "vector_dim": meta.vector_dim,
            "created_at": meta.created_at,
            "updated_at": meta.updated_at,
            "endpoint": {
                "base_url": self._config.endpoint.base_url,
                "llm_model": self._config.endpoint.llm_model,
                "embedding_model": self._config.endpoint.embedding_model,
                "min_score": self._config.retrieval.min_score,
            },
        }

    def ask_stream(self, question: str) -> Iterator[dict[str, Any]]:
        """Yield answer events: tokens as they stream, then a final result.

        The events are plain dicts so the HTTP layer can serialize them to
        server-sent events without knowing any engine types.
        """

        from ..core.errors import ModelUnavailableError

        with self._lock:
            index = self._index
            engine = self._engine
            llm = self._llm

        try:
            prep = engine.prepare(index, question)
        except ModelUnavailableError as exc:
            yield {"type": "error", "message": str(exc)}
            return

        if not prep.found:
            yield {"type": "done", "answer": self._serialize_answer(_not_found())}
            return

        pieces: list[str] = []
        try:
            for token in llm.stream(prep.messages):
                pieces.append(token)
                yield {"type": "token", "text": token}
        except ModelUnavailableError as exc:
            yield {"type": "error", "message": str(exc)}
            return

        answer = engine.assemble(prep, "".join(pieces))
        yield {"type": "done", "answer": self._serialize_answer(answer)}

    def add_path(self, path: str) -> dict[str, Any]:
        """Ingest a file or folder already on disk, then return the new info."""

        with self._lock:
            self._index, _summary = self._pipeline.update(
                self._index,
                [path],
                embedding_model=self._embedding_model,
                timestamp=_now(),
            )
            self._persist()
        return self.info()

    def add_upload(self, filename: str, data: bytes) -> dict[str, Any]:
        """Save uploaded bytes next to the index, then ingest the saved file."""

        safe_name = os.path.basename(filename) or "upload"
        index_dir = os.path.dirname(self._index_path) or "."
        target = os.path.join(index_dir, "uploads", safe_name)
        self._fs.write_bytes(target, data)
        return self.add_path(target)

    def remove(self, target: str) -> dict[str, Any]:
        """Drop a document from the index by path or file name."""

        with self._lock:
            self._index, _summary = self._pipeline.remove(
                self._index, [target], timestamp=_now()
            )
            self._persist()
        return self.info()

    def page(self) -> str:
        """The bundled single-page front end."""

        from importlib.resources import files

        return (files("cogwright.web") / "static" / "index.html").read_text(
            encoding="utf-8"
        )

    def _load_index(self) -> Index:
        if self._fs.exists(self._index_path):
            return Index.from_dict(json.loads(self._fs.read_text(self._index_path)))
        return Index.build(
            [],
            InMemoryVectorStore(),
            IndexMetadata(embedding_model=self._embedding_model),
        )

    def _persist(self) -> None:
        self._fs.write_text(self._index_path, json.dumps(self._index.to_dict()))

    def _serialize_answer(self, answer: Answer) -> dict[str, Any]:
        seen: set[tuple[str, int, str | None]] = set()
        citations: list[dict[str, Any]] = []
        for citation in answer.citations:
            key = (citation.source_path, citation.page, citation.section)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "source_path": citation.source_path,
                    "page": citation.page,
                    "section": citation.section,
                }
            )
        text = (
            clean_answer_text(answer.text, NOT_FOUND_MESSAGE)
            if answer.found
            else answer.text
        )
        return {
            "found": answer.found,
            "text": text,
            "codes": [code.value for code in answer.referenced_codes],
            "citations": citations,
            "retrieved": [
                {
                    "chunk_id": sc.chunk.chunk_id,
                    "page": sc.chunk.page,
                    "section": sc.chunk.section,
                    "score": round(sc.score, 3),
                    "match_type": sc.match_type,
                }
                for sc in answer.retrieved
            ],
        }


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _not_found() -> Answer:
    return Answer(text=NOT_FOUND_MESSAGE, found=False)
