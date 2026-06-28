# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Orchestration: wiring the pure stages together behind the protocol seams.

:class:`IngestionPipeline` parses a corpus into a persisted :class:`Index`.
:class:`QueryEngine` answers a question against an index. Both take their I/O as
injected protocols (filesystem, parsers, embedder, model), so the whole flow is
exercised in tests with fakes and never needs a live endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .chunking import chunk_document, embedding_text
from .citation import CitationMapper
from .code_index import CodeIndexer
from .config import Config
from .errors import UnsupportedDocumentError
from .index import Index
from .models import Answer, Chunk, CodeRef, Document, Message, ScoredChunk
from .prompt import NOT_FOUND_MESSAGE, PromptBuilder
from .protocols import DocumentParser, Embedder, FileSystem, LLMClient
from .retrieval import retrieve
from .vector_store import InMemoryVectorStore

# Extensions worth scanning when a corpus path is a directory. Each candidate is
# still confirmed against a parser's supports() before being read.
_DEFAULT_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".text", ".pdf")


class IngestionPipeline:
    """Builds an index from a corpus of documents."""

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
        parser = self._parser_for(path)
        if parser is None:
            raise UnsupportedDocumentError(path)
        return parser.parse(path, self._fs.read_bytes(path))

    def chunk_documents(self, documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(
                chunk_document(document, self._config.chunking, self._indexer)
            )
        return chunks

    def build_index(self, chunks: Sequence[Chunk]) -> Index:
        store = InMemoryVectorStore()
        if chunks:
            vectors = self._embedder.embed([embedding_text(c) for c in chunks])
            for chunk, vector in zip(chunks, vectors):
                store.add(chunk.chunk_id, vector)
        return Index.build(list(chunks), store)

    def ingest(self, corpus_paths: Sequence[str]) -> Index:
        documents = [self.parse_file(path) for path in self.collect_files(corpus_paths)]
        chunks = self.chunk_documents(documents)
        return self.build_index(chunks)

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
    ) -> None:
        self._embedder = embedder
        self._llm = llm
        self._config = config
        self._indexer = CodeIndexer(config.code_patterns)
        self._prompt = prompt_builder or PromptBuilder()

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

        normalized = answer_text.strip()
        if not prep.found or normalized == NOT_FOUND_MESSAGE:
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
