# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Command-line entry point: ``ingest`` and ``ask``.

This layer only wires real I/O (the disk, the parsers, the configured endpoint)
to the pure engine and formats results. It keeps no logic of its own beyond
argument handling and printing, and it never crashes on an unreachable endpoint;
it reports the problem and exits non-zero.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Sequence

from ..adapters.filesystem import RealFileSystem
from ..adapters.http_endpoint import HttpEmbedder, HttpLLMClient
from ..adapters.ocr import PytesseractOcrEngine
from ..adapters.pdf_parser import PdfDocumentParser
from ..adapters.text_parser import TextDocumentParser
from ..adapters.vision import VisionDiagramAnalyzer
from ..core.citation import clean_answer_text
from ..core.config import Config, EndpointConfig, RetrievalConfig
from ..core.engine import IngestionPipeline, QueryEngine
from ..core.errors import CogwrightError, ModelUnavailableError
from ..core.index import Index
from ..core.models import Answer, ScoredChunk
from ..core.prompt import NOT_FOUND_MESSAGE
from ..core.protocols import DocumentParser
from ..eval import evaluate, parse_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        if args.command == "ingest":
            return _cmd_ingest(args)
        if args.command == "ask":
            return _cmd_ask(args)
        if args.command == "eval":
            return _cmd_eval(args)
    except ModelUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "The configured endpoint could not be reached. Check that it is "
            "running and that --base-url points at it.",
            file=sys.stderr,
        )
        return 3
    except CogwrightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cogwright",
        description=(
            "Local-first retrieval engine for technical equipment documentation."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser(
        "ingest", help="parse a corpus of documents and build a persisted index"
    )
    ingest.add_argument("paths", nargs="+", help="files or directories to ingest")
    _add_common_args(ingest)
    ingest.add_argument(
        "--embedding-model",
        default=os.environ.get("COGWRIGHT_EMBEDDING_MODEL"),
        help="embedding model name served by the endpoint",
    )
    ingest.add_argument(
        "--ocr",
        action="store_true",
        help="recognize scanned PDF pages that have no text layer (needs the ocr extra)",
    )
    ingest.add_argument(
        "--diagrams",
        action="store_true",
        help="transcribe diagram callouts in PDFs with a multimodal vision model",
    )
    ingest.add_argument(
        "--vision-model",
        default=os.environ.get("COGWRIGHT_VISION_MODEL"),
        help="multimodal model name for diagram analysis",
    )

    ask = subparsers.add_parser(
        "ask", help="ask a question and get a grounded, cited answer"
    )
    ask.add_argument("question", help="the natural-language question")
    _add_common_args(ask)
    ask.add_argument(
        "--embedding-model",
        default=os.environ.get("COGWRIGHT_EMBEDDING_MODEL"),
        help="embedding model name served by the endpoint",
    )
    ask.add_argument(
        "--llm-model",
        default=os.environ.get("COGWRIGHT_LLM_MODEL"),
        help="chat model name served by the endpoint",
    )
    ask.add_argument("--top-k", type=int, default=None, help="passages to retrieve")
    ask.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="minimum cosine score for a semantic match (calibrate per model)",
    )
    ask.add_argument(
        "--no-stream", action="store_true", help="wait for the full answer"
    )
    ask.add_argument(
        "--show-retrieved",
        action="store_true",
        help="print the retrieved passages and their scores before answering",
    )
    ask.add_argument(
        "--json",
        action="store_true",
        help="ask the model for a structured JSON answer (reliable steps and "
        "citations; best with a capable model, falls back to prose otherwise)",
    )

    evaluate_cmd = subparsers.add_parser(
        "eval",
        help="score retrieval quality against a graded dataset (no model call)",
    )
    evaluate_cmd.add_argument("dataset", help="path to a graded dataset JSON file")
    _add_common_args(evaluate_cmd)
    evaluate_cmd.add_argument(
        "--embedding-model",
        default=os.environ.get("COGWRIGHT_EMBEDDING_MODEL"),
        help="embedding model name served by the endpoint",
    )
    evaluate_cmd.add_argument("--top-k", type=int, default=None, help="passages to retrieve")
    evaluate_cmd.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="minimum cosine score for a semantic match (calibrate per model)",
    )
    return parser


def _add_common_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--index",
        default=os.environ.get("COGWRIGHT_INDEX", ".cogwright/index.json"),
        help="path to the persisted index file",
    )
    sub.add_argument(
        "--base-url",
        default=os.environ.get("COGWRIGHT_BASE_URL"),
        help="base URL of the OpenAI-compatible endpoint",
    )
    sub.add_argument(
        "--api-key",
        default=os.environ.get("COGWRIGHT_API_KEY"),
        help="API key for the endpoint, if it requires one",
    )


def _config_from_args(args: argparse.Namespace) -> Config:
    defaults = EndpointConfig()
    endpoint = EndpointConfig(
        base_url=args.base_url or defaults.base_url,
        api_key=args.api_key,
        llm_model=getattr(args, "llm_model", None) or defaults.llm_model,
        embedding_model=args.embedding_model or defaults.embedding_model,
        vision_model=getattr(args, "vision_model", None) or defaults.vision_model,
    )
    retrieval = RetrievalConfig()
    top_k = getattr(args, "top_k", None)
    if top_k:
        retrieval = dataclasses.replace(retrieval, top_k=top_k)
    min_score = getattr(args, "min_score", None)
    if min_score is not None:
        retrieval = dataclasses.replace(retrieval, min_score=min_score)
    return Config(index_path=args.index, endpoint=endpoint, retrieval=retrieval)


def _make_embedder(config: Config) -> HttpEmbedder:
    endpoint = config.endpoint
    return HttpEmbedder(
        base_url=endpoint.base_url,
        model=endpoint.embedding_model,
        api_key=endpoint.api_key,
        timeout=endpoint.timeout_seconds,
    )


def _make_llm(config: Config) -> HttpLLMClient:
    endpoint = config.endpoint
    return HttpLLMClient(
        base_url=endpoint.base_url,
        model=endpoint.llm_model,
        api_key=endpoint.api_key,
        timeout=endpoint.timeout_seconds,
    )


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    fs = RealFileSystem()
    ocr_engine = PytesseractOcrEngine() if args.ocr else None
    diagram_analyzer = (
        VisionDiagramAnalyzer(
            base_url=config.endpoint.base_url,
            model=config.endpoint.vision_model,
            api_key=config.endpoint.api_key,
            timeout=config.endpoint.timeout_seconds,
        )
        if args.diagrams
        else None
    )
    parsers: list[DocumentParser] = [
        TextDocumentParser(),
        PdfDocumentParser(ocr_engine=ocr_engine, diagram_analyzer=diagram_analyzer),
    ]
    embedder = _make_embedder(config)

    pipeline = IngestionPipeline(fs, parsers, embedder, config)
    files = pipeline.collect_files(args.paths)
    if not files:
        print("error: no supported documents found in the given paths.", file=sys.stderr)
        return 1

    print(f"Parsing {len(files)} document(s)...")
    documents = [pipeline.parse_file(path) for path in files]
    chunks = pipeline.chunk_documents(documents)
    print(f"Built {len(chunks)} chunk(s); embedding against the endpoint...")
    index = pipeline.build_index(chunks)

    fs.write_text(config.index_path, json.dumps(index.to_dict()))
    codes = sorted(index.code_index.values)
    print(f"Indexed {len(index.chunks)} chunk(s) to {config.index_path}")
    if codes:
        preview = ", ".join(codes[:12])
        suffix = ", ..." if len(codes) > 12 else ""
        print(f"Resolvable identifiers ({len(codes)}): {preview}{suffix}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    fs = RealFileSystem()
    if not fs.exists(config.index_path):
        print(
            f"error: no index found at {config.index_path}. "
            "Run 'cogwright ingest <paths>' first.",
            file=sys.stderr,
        )
        return 1

    index = Index.from_dict(json.loads(fs.read_text(config.index_path)))
    embedder = _make_embedder(config)
    llm = _make_llm(config)
    engine = QueryEngine(embedder, llm, config, structured=args.json)

    prep = engine.prepare(index, args.question)
    if args.show_retrieved:
        _print_retrieved(prep.retrieved)
    if not prep.found:
        print(NOT_FOUND_MESSAGE)
        return 0

    # A JSON reply cannot be streamed legibly, so it is collected in full.
    if args.no_stream or args.json:
        answer_text = llm.complete(prep.messages)
    else:
        chunks: list[str] = []
        for piece in llm.stream(prep.messages):
            chunks.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
        answer_text = "".join(chunks)

    answer = engine.assemble(prep, answer_text)
    if args.no_stream or args.json:
        # Streaming already printed the raw text live; for the buffered and JSON
        # paths show the rendered answer, markers and any stray not-found line
        # removed. A not-found JSON reply collapses to the standard message.
        if not answer.found:
            print(NOT_FOUND_MESSAGE)
        else:
            print(clean_answer_text(answer.text, NOT_FOUND_MESSAGE))
    _print_provenance(answer)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    fs = RealFileSystem()
    if not fs.exists(config.index_path):
        print(
            f"error: no index found at {config.index_path}. "
            "Run 'cogwright ingest <paths>' first.",
            file=sys.stderr,
        )
        return 1
    if not fs.exists(args.dataset):
        print(f"error: no dataset found at {args.dataset}.", file=sys.stderr)
        return 1

    index = Index.from_dict(json.loads(fs.read_text(config.index_path)))
    cases = parse_dataset(json.loads(fs.read_text(args.dataset)))
    engine = QueryEngine(_make_embedder(config), _make_llm(config), config)

    report = evaluate(cases, lambda question: engine.prepare(index, question))
    print(report.summary())
    return 0


def _print_retrieved(retrieved: Sequence[ScoredChunk]) -> None:
    if not retrieved:
        print("Retrieved: (nothing cleared the relevance threshold)")
        return
    print("Retrieved:")
    for scored in retrieved:
        chunk = scored.chunk
        label = f"  [{chunk.chunk_id}] p{chunk.page} {scored.match_type} {scored.score:.3f}"
        if chunk.section:
            label += f" | {chunk.section}"
        print(label)
    print()


def _print_provenance(answer: Answer) -> None:
    if answer.referenced_codes:
        identifiers = ", ".join(code.value for code in answer.referenced_codes)
        print(f"\nReferenced identifiers: {identifiers}")
    if answer.citations:
        print("\nSources:")
        seen: set[tuple[str, int, str | None]] = set()
        for citation in answer.citations:
            key = (citation.source_path, citation.page, citation.section)
            if key in seen:
                continue
            seen.add(key)
            location = f"page {citation.page}"
            if citation.section:
                location += f", section: {citation.section}"
            print(f"  - {citation.source_path} ({location})")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
