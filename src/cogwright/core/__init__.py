# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""The pure core: retrieval and decision logic behind protocol seams.

Nothing in this package imports a specific model, vector database, or web
framework. Concrete I/O lives in :mod:`cogwright.adapters`.
"""

from __future__ import annotations

from .chunking import chunk_document, embedding_text
from .citation import CitationMapper, clean_answer_text, strip_citation_markers
from .code_index import CodeIndex, CodeIndexer
from .config import (
    DEFAULT_CODE_PATTERNS,
    ChunkingConfig,
    CodePattern,
    Config,
    EndpointConfig,
    RetrievalConfig,
)
from .engine import (
    IngestionPipeline,
    IngestSummary,
    Preparation,
    QueryEngine,
    not_found_answer,
)
from .errors import (
    CogwrightError,
    ModelUnavailableError,
    UnsupportedDocumentError,
)
from .index import DocumentRecord, Index, IndexMetadata
from .models import (
    Answer,
    BlockKind,
    Chunk,
    Citation,
    CodeRef,
    Document,
    Message,
    ScoredChunk,
    TextBlock,
    Vector,
)
from .prompt import NOT_FOUND_MESSAGE, STRUCTURED_SYSTEM_PROMPT, PromptBuilder
from .protocols import (
    DiagramAnalyzer,
    DocumentParser,
    Embedder,
    FileSystem,
    LLMClient,
    OcrEngine,
    VectorStore,
)
from .retrieval import retrieve
from .structured import StructuredAnswer, parse_structured, render_answer_text
from .vector_store import InMemoryVectorStore, cosine_similarity

__all__ = [
    "Answer",
    "BlockKind",
    "Chunk",
    "ChunkingConfig",
    "Citation",
    "CitationMapper",
    "CodeIndex",
    "CodeIndexer",
    "CodePattern",
    "CodeRef",
    "CogwrightError",
    "Config",
    "DEFAULT_CODE_PATTERNS",
    "DiagramAnalyzer",
    "Document",
    "DocumentRecord",
    "DocumentParser",
    "Embedder",
    "EndpointConfig",
    "FileSystem",
    "InMemoryVectorStore",
    "Index",
    "IndexMetadata",
    "IngestSummary",
    "IngestionPipeline",
    "LLMClient",
    "Message",
    "ModelUnavailableError",
    "NOT_FOUND_MESSAGE",
    "OcrEngine",
    "Preparation",
    "PromptBuilder",
    "QueryEngine",
    "RetrievalConfig",
    "STRUCTURED_SYSTEM_PROMPT",
    "ScoredChunk",
    "StructuredAnswer",
    "TextBlock",
    "UnsupportedDocumentError",
    "Vector",
    "VectorStore",
    "chunk_document",
    "clean_answer_text",
    "cosine_similarity",
    "embedding_text",
    "not_found_answer",
    "parse_structured",
    "render_answer_text",
    "retrieve",
    "strip_citation_markers",
]
