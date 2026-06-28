# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Configuration structures for the retrieval engine.

Everything that tunes behavior lives here so the pipeline stages stay pure and
take their settings as data. Endpoint defaults point at a model server running
locally; the model names are placeholders meant to be set to whatever the chosen
endpoint serves.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CodePattern:
    """A configurable rule for detecting an identifier in text.

    ``regex`` must expose a named group ``id`` capturing the identifier portion.
    The normalized lookup key is ``f"{canonical_prefix}-{id}"`` upper-cased, so a
    bare ``alarm 204`` and ``AL204`` both resolve to ``AL-204``.
    """

    name: str
    canonical_prefix: str
    regex: str


# Generic, industry-neutral identifier rules. They cover the common shapes of
# alarm, stop, fault, error, and part identifiers found in equipment manuals
# without naming any vendor scheme. Override or extend through Config.
DEFAULT_CODE_PATTERNS: tuple[CodePattern, ...] = (
    CodePattern(
        name="alarm",
        canonical_prefix="AL",
        regex=r"(?:ALARM|ALM|AL)[\s\-:_]*(?P<id>\d{1,4}[A-Z]?)",
    ),
    CodePattern(
        name="stop",
        canonical_prefix="SC",
        regex=r"(?:STOP[\s\-]?CODE|STOP|SC)[\s\-:_]*(?P<id>\d{1,4})",
    ),
    CodePattern(
        name="fault",
        canonical_prefix="F",
        regex=r"(?:FAULT|FLT|F)[\s\-:_]*(?P<id>\d{2,4})",
    ),
    CodePattern(
        name="error",
        canonical_prefix="E",
        regex=r"(?:ERROR|ERR|E)[\s\-:_]*(?P<id>\d{2,4})",
    ),
    CodePattern(
        name="part",
        canonical_prefix="PN",
        regex=(
            r"(?:PART\s+(?:NO\.?|NUMBER)|PART|P/?N)"
            r"[\s\-:#]*(?P<id>[A-Z0-9]{2,}(?:-[A-Z0-9]+)+)"
        ),
    ),
)


@dataclass(frozen=True)
class ChunkingConfig:
    """Controls how parsed documents are split into retrievable chunks."""

    max_chars: int = 1200
    keep_tables_intact: bool = True
    keep_steps_intact: bool = True


@dataclass(frozen=True)
class RetrievalConfig:
    """Controls how candidate chunks are selected and ranked."""

    top_k: int = 6
    # A semantic match below this cosine score is treated as not relevant. When
    # no chunk clears the bar and no exact code lookup hits, the engine returns
    # the not-found answer rather than inventing one.
    min_score: float = 0.15
    # Exact code and part lookups are precise, so they are ranked above any
    # purely semantic match. This base score is added on top of cosine.
    code_base_score: float = 1.0


@dataclass(frozen=True)
class EndpointConfig:
    """Where to reach the chat-completions and embeddings HTTP endpoint.

    Defaults target a model server on the local machine. No network call is made
    except to this user-configured endpoint. The model names are placeholders;
    set them to the names your endpoint exposes via the CLI flags or environment.
    """

    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    llm_model: str = "local-chat-model"
    embedding_model: str = "local-embedding-model"
    timeout_seconds: float = 60.0


@dataclass(frozen=True)
class Config:
    """Top-level configuration for ingestion and querying."""

    corpus_paths: tuple[str, ...] = ()
    index_path: str = ".cogwright/index.json"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    code_patterns: tuple[CodePattern, ...] = DEFAULT_CODE_PATTERNS
