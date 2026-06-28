# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for the command-line layer, including graceful endpoint failures.

A guaranteed-unreachable endpoint (the discard port) stands in for a down model
server, so the not-crash behavior is verified with no network dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

from cogwright.adapters.filesystem import RealFileSystem
from cogwright.adapters.text_parser import TextDocumentParser
from cogwright.cli.main import main
from cogwright.core.config import Config
from cogwright.core.engine import IngestionPipeline

from .fakes import FakeEmbedder

FIXTURE = Path(__file__).parent / "fixtures" / "sample_manual" / "series7_conveyor_manual.txt"
UNREACHABLE = "http://127.0.0.1:9/v1"  # discard port: refuses connections immediately


def _write_index(path: Path) -> None:
    pipeline = IngestionPipeline(
        RealFileSystem(), [TextDocumentParser()], FakeEmbedder(), Config()
    )
    index = pipeline.ingest([str(FIXTURE)])
    RealFileSystem().write_text(str(path), json.dumps(index.to_dict()))


def test_no_command_prints_help_and_returns_two() -> None:
    assert main([]) == 2


def test_ask_without_index_reports_cleanly() -> None:
    code = main(["ask", "anything", "--index", "/no/such/index.json"])
    assert code == 1


def test_ingest_against_unreachable_endpoint_exits_three(tmp_path: Path) -> None:
    code = main(
        [
            "ingest",
            str(FIXTURE),
            "--index",
            str(tmp_path / "index.json"),
            "--base-url",
            UNREACHABLE,
        ]
    )
    assert code == 3


def test_ask_against_unreachable_endpoint_exits_three(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path)
    code = main(
        [
            "ask",
            "How do I clear alarm 204?",
            "--index",
            str(index_path),
            "--base-url",
            UNREACHABLE,
        ]
    )
    assert code == 3
