# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Orchestration: wiring the pure stages together behind the protocol seams.

:class:`IngestionPipeline` parses a corpus into a persisted :class:`Index`.
:class:`QueryEngine` answers a question against an index. Both take their I/O as
injected protocols (filesystem, parsers, embedder, model), so the whole flow is
exercised in tests with fakes and never needs a live endpoint.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .chunking import chunk_document, embedding_text
from .citation import CitationMapper, strip_citation_markers
from .code_index import CodeIndexer
from .config import Config
from .errors import CogwrightError, UnsupportedDocumentError
from .index import DocumentRecord, Index, IndexMetadata
from .models import Answer, Chunk, CodeRef, Document, Message, ScoredChunk, Vector
from .prompt import NOT_FOUND_MESSAGE, STRUCTURED_SYSTEM_PROMPT, PromptBuilder
from .protocols import DocumentParser, Embedder, FileSystem, LLMClient
from .retrieval import retrieve
from .structured import parse_structured, render_answer_text
from .vector_store import InMemoryVectorStore

# Extensions worth scanning when a corpus path is a directory. Each candidate is
# still confirmed against a parser's supports() before being read.
_DEFAULT_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".text", ".pdf")


@dataclass(frozen=True)
class IngestSummary:
    """What a build, update, or remove operation changed."""

    added: tuple[str, ...] = ()
    refreshed: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    total_documents: int = 0
    total_chunks: int = 0


class IngestionPipeline:
    """Builds and maintains an index over a corpus of documents."""

    def __init__(
        self,
        fs: FileSystem,
        parsers: Sequence[DocumentParser],
        embedder: Embedder,
        config: Config,
    ) -> None:
        self._fs = fs
        self._parsers = tuple(parsers)
        self._embedder = embedder
        self._config = config
        self._indexer = CodeIndexer(config.code_patterns)

    def collect_files(self, corpus_paths: Sequence[str]) -> list[str]:
        """Resolve corpus paths (files or directories) to supported files."""

        files: list[str] = []
        seen: set[str] = set()
        for path in corpus_paths:
            candidates = (
                [path]
                if self._is_file(path)
                else self._fs.list_files(path, _DEFAULT_EXTENSIONS)
            )
            for candidate in candidates:
                if candidate in seen:
                    continue
                if self._parser_for(candidate) is not None:
                    seen.add(candidate)
                    files.append(candidate)
        return files

    def parse_file(self, path: str) -> Document:
        return self._parse(path, self._fs.read_bytes(path))

    def chunk_documents(self, documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(
                chunk_document(document, self._config.chunking, self._indexer)
            )
        return chunks

    def ingest(
        self,
        corpus_paths: Sequence[str],
        *,
        embedding_model: str = "",
        timestamp: str = "",
    ) -> Index:
        """Build a fresh index over the corpus, stamping it with metadata."""

        index = Index.build(
            [],
            InMemoryVectorStore(),
            IndexMetadata(
                embedding_model=embedding_model,
                created_at=timestamp,
                updated_at=timestamp,
            ),
        )
        index, _summary = self.update(
            index, corpus_paths, embedding_model=embedding_model, timestamp=timestamp
        )
        return index

    def update(
        self,
        index: Index,
        corpus_paths: Sequence[str],
        *,
        embedding_model: str = "",
        timestamp: str = "",
    ) -> tuple[Index, IngestSummary]:
        """Add new documents and refresh changed ones, leaving the rest untouched."""

        self._require_matching_model(index, embedding_model)
        records = {d.source_path: d for d in index.metadata.documents}
        added: list[str] = []
        refreshed: list[str] = []
        skipped: list[str] = []

        for path in self.collect_files(corpus_paths):
            data = self._fs.read_bytes(path)
            digest = _content_hash(data)
            existing = records.get(path)
            if existing is not None and existing.content_hash == digest:
                skipped.append(path)
                continue
            if existing is not None:
                index.remove_document(path)
                refreshed.append(path)
            else:
                added.append(path)
            document = self._parse(path, data)
            chunks = chunk_document(document, self._config.chunking, self._indexer)
            index.add_chunks(chunks, self._embed(chunks))
            records[path] = DocumentRecord(
                source_path=path,
                document_id=document.document_id,
                title=document.title,
                content_hash=digest,
                chunk_count=len(chunks),
            )

        index.metadata = _refreshed_metadata(
            index, embedding_model, tuple(records.values()), timestamp
        )
        return index, IngestSummary(
            added=tuple(added),
            refreshed=tuple(refreshed),
            skipped=tuple(skipped),
            total_documents=len(records),
            total_chunks=len(index.chunks),
        )

    def remove(
        self,
        index: Index,
        targets: Sequence[str],
        *,
        timestamp: str = "",
    ) -> tuple[Index, IngestSummary]:
        """Drop documents from the index, matched by path or file name."""

        records = {d.source_path: d for d in index.metadata.documents}
        removed: list[str] = []
        for target in targets:
            for path in _resolve_targets(records.keys(), target):
                index.remove_document(path)
                del records[path]
                removed.append(path)

        index.metadata = _refreshed_metadata(
            index,
            index.metadata.embedding_model,
            tuple(records.values()),
            timestamp,
            created_at=index.metadata.created_at,
        )
        return index, IngestSummary(
            removed=tuple(removed),
            total_documents=len(records),
            total_chunks=len(index.chunks),
        )

    def _embed(self, chunks: Sequence[Chunk]) -> list[Vector]:
        if not chunks:
            return []
        return self._embedder.embed([embedding_text(c) for c in chunks])

    def _parse(self, path: str, data: bytes) -> Document:
        parser = self._parser_for(path)
        if parser is None:
            raise UnsupportedDocumentError(path)
        return parser.parse(path, data)

    def _require_matching_model(self, index: Index, embedding_model: str) -> None:
        existing = index.metadata.embedding_model
        if existing and embedding_model and existing != embedding_model:
            raise CogwrightError(
                f"index was built with embedding model {existing!r}, but "
                f"{embedding_model!r} was given. Rebuild the index, or pass the "
                "model the index was built with."
            )

    def _parser_for(self, path: str) -> DocumentParser | None:
        for parser in self._parsers:
            if parser.supports(path):
                return parser
        return None

    def _is_file(self, path: str) -> bool:
        # A path is treated as a file when it has a known document extension or
        # exists as a non-directory. Directories are expanded via list_files.
        lowered = path.lower()
        if any(lowered.endswith(ext) for ext in _DEFAULT_EXTENSIONS):
            return True
        return self._fs.exists(path) and not self._fs.list_files(path, ())


@dataclass(frozen=True)
class Preparation:
    """Everything decided before the model is called for one query."""

    query: str
    found: bool
    retrieved: tuple[ScoredChunk, ...]
    resolved_codes: tuple[CodeRef, ...]
    messages: tuple[Message, ...]


class QueryEngine:
    """Answers questions against an index, grounded and cited."""

    def __init__(
        self,
        embedder: Embedder,
        llm: LLMClient,
        config: Config,
        prompt_builder: PromptBuilder | None = None,
        structured: bool = False,
    ) -> None:
        self._embedder = embedder
        self._llm = llm
        self._config = config
        self._indexer = CodeIndexer(config.code_patterns)
        self._structured = structured
        if prompt_builder is not None:
            self._prompt = prompt_builder
        elif structured:
            self._prompt = PromptBuilder(system_prompt=STRUCTURED_SYSTEM_PROMPT)
        else:
            self._prompt = PromptBuilder()

    def prepare(self, index: Index, query: str) -> Preparation:
        """Run retrieval and decide whether the question can be grounded."""

        query_codes = self._indexer.extract(query)
        query_vector = self._embedder.embed([query])[0]
        retrieved = retrieve(
            index, query_vector, query_codes, self._config.retrieval
        )
        resolved = tuple(
            code for code in query_codes if code.value in index.code_index.values
        )
        found = len(retrieved) > 0
        messages = (
            tuple(self._prompt.build(query, retrieved)) if found else ()
        )
        return Preparation(
            query=query,
            found=found,
            retrieved=tuple(retrieved),
            resolved_codes=resolved,
            messages=messages,
        )

    def assemble(self, prep: Preparation, answer_text: str) -> Answer:
        """Turn raw model output into a structured, cited answer."""

        if not prep.found:
            return not_found_answer()
        if self._structured:
            structured = self._assemble_structured(prep, answer_text)
            if structured is not None:
                return structured
            # The model did not produce usable JSON; fall back to prose.
        return self._assemble_prose(prep, answer_text)

    def _assemble_prose(self, prep: Preparation, answer_text: str) -> Answer:
        normalized = answer_text.strip()
        if _is_not_found(normalized):
            return not_found_answer()

        chunk_map = {sc.chunk.chunk_id: sc.chunk for sc in prep.retrieved}
        mapper = CitationMapper(chunk_map)
        citations = mapper.map_answer(normalized, prep.retrieved)
        referenced = _referenced_codes(prep, chunk_map)
        return Answer(
            text=normalized,
            found=True,
            citations=citations,
            referenced_codes=referenced,
            retrieved=prep.retrieved,
        )

    def _assemble_structured(
        self, prep: Preparation, answer_text: str
    ) -> Answer | None:
        chunk_map = {sc.chunk.chunk_id: sc.chunk for sc in prep.retrieved}
        parsed = parse_structured(answer_text, chunk_map.keys())
        if parsed is None:
            return None
        # An answer with no content is not-found. The model's "found" flag is not
        # trusted on its own: smaller models set it to false while still returning
        # steps, so the presence of content wins over the flag.
        if parsed.is_empty:
            return not_found_answer()

        mapper = CitationMapper(chunk_map)
        if parsed.used_ids:
            citations = tuple(
                citation
                for citation in (mapper.citation_for(cid) for cid in parsed.used_ids)
                if citation is not None
            )
        else:
            citations = mapper.map_answer("", prep.retrieved)
        return Answer(
            text=render_answer_text(parsed),
            found=True,
            citations=citations,
            referenced_codes=_referenced_codes(prep, chunk_map),
            retrieved=prep.retrieved,
        )

    def ask(self, index: Index, query: str) -> Answer:
        """Convenience path: prepare, call the model, assemble. No streaming."""

        prep = self.prepare(index, query)
        if not prep.found:
            return not_found_answer()
        answer_text = self._llm.complete(prep.messages)
        return self.assemble(prep, answer_text)


def not_found_answer() -> Answer:
    """The clean, grounded-failure result used on every not-found path."""

    return Answer(text=NOT_FOUND_MESSAGE, found=False)


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _refreshed_metadata(
    index: Index,
    embedding_model: str,
    documents: tuple[DocumentRecord, ...],
    updated_at: str,
    created_at: str | None = None,
) -> IndexMetadata:
    base = index.metadata
    return IndexMetadata(
        embedding_model=embedding_model or base.embedding_model,
        vector_dim=_dim_of(index),
        created_at=created_at if created_at is not None else (base.created_at or updated_at),
        updated_at=updated_at,
        documents=documents,
    )


def _dim_of(index: Index) -> int | None:
    for vector in index.store.vectors().values():
        return len(vector)
    return None


def _resolve_targets(paths: Iterable[str], target: str) -> list[str]:
    available = list(paths)
    if target in available:
        return [target]
    base = os.path.basename(target)
    return [p for p in available if os.path.basename(p) == base or p.endswith(target)]


# How much extra text around the not-found sentence still counts as a not-found
# reply, allowing a short lead-in like "Unfortunately," without matching a real
# answer that only mentions the phrase.
_NOT_FOUND_SLACK_CHARS = 30


def _is_not_found(answer_text: str) -> bool:
    """Whether a model reply is really the not-found sentence.

    Real models add stray whitespace, a trailing period, a citation, or a short
    lead-in like "Unfortunately,". This recognizes those without firing on a
    genuine answer that merely mentions the phrase in passing.
    """

    core = strip_citation_markers(answer_text).strip().lower().rstrip(".!").strip()
    target = NOT_FOUND_MESSAGE.lower().rstrip(".").strip()
    if not core:
        return False
    if core == target:
        return True
    return target in core and len(core) <= len(target) + _NOT_FOUND_SLACK_CHARS


def _referenced_codes(
    prep: Preparation,
    chunk_map: dict[str, Chunk],
) -> tuple[CodeRef, ...]:
    """Identifiers worth surfacing for the answer.

    When the question itself named identifiers that resolved in the corpus, those
    are the precise references and the only ones surfaced. Otherwise fall back to
    the identifiers present in the retrieved passages, so a parts lookup still
    surfaces the relevant part numbers.
    """

    if prep.resolved_codes:
        return prep.resolved_codes

    result: list[CodeRef] = []
    seen: set[str] = set()
    for chunk in chunk_map.values():
        for code in chunk.codes:
            if code.value not in seen:
                seen.add(code.value)
                result.append(code)
    return tuple(result)
