# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Tests for the web app logic and the HTTP server, using fakes throughout."""

from __future__ import annotations

import json
import threading
import urllib.request
from collections.abc import Iterator
from functools import partial
from http.server import ThreadingHTTPServer

import pytest

from cogwright.adapters.text_parser import TextDocumentParser
from cogwright.core.config import Config
from cogwright.web.app import WebApp
from cogwright.web.server import _Handler

from .fakes import FakeEmbedder, FakeFileSystem, FakeLLMClient

ALARM = (
    "ALARM REFERENCE\n\nAlarm 204 means low coolant pressure.\n\n"
    "1. Stop the unit and let it cool.\n2. Clear alarm 204 and restart.\n"
)


def _app(fs: FakeFileSystem, llm: FakeLLMClient | None = None) -> WebApp:
    return WebApp(
        Config(),
        "index.json",
        fs,
        [TextDocumentParser()],
        FakeEmbedder(),
        llm or FakeLLMClient(),
    )


def test_info_starts_empty() -> None:
    app = _app(FakeFileSystem())
    info = app.info()
    assert info["document_count"] == 0
    assert info["chunk_count"] == 0
    assert "min_score" in info["endpoint"]


def test_add_path_indexes_and_persists() -> None:
    fs = FakeFileSystem()
    fs.add_text("manual.txt", ALARM)
    app = _app(fs)

    info = app.add_path("manual.txt")
    assert info["document_count"] == 1
    assert info["chunk_count"] >= 1
    # The index was written back to disk.
    assert fs.exists("index.json")


def test_ask_streams_then_returns_a_cited_answer() -> None:
    fs = FakeFileSystem()
    fs.add_text("manual.txt", ALARM)
    app = _app(fs)
    app.add_path("manual.txt")

    events = list(app.ask_stream("How do I clear alarm 204?"))

    assert any(e["type"] == "token" for e in events)
    done = events[-1]
    assert done["type"] == "done"
    answer = done["answer"]
    assert answer["found"] is True
    assert "1." in answer["text"]
    assert "AL-204" in answer["codes"]
    assert answer["citations"]
    assert answer["retrieved"]


def test_ask_unanswerable_is_not_found_without_calling_model() -> None:
    fs = FakeFileSystem()
    fs.add_text("manual.txt", ALARM)
    llm = FakeLLMClient()
    app = _app(fs, llm=llm)
    app.add_path("manual.txt")

    events = list(app.ask_stream("Bluetooth pairing instructions"))
    done = events[-1]

    assert done["type"] == "done"
    assert done["answer"]["found"] is False
    assert llm.calls == []


def test_remove_drops_the_document() -> None:
    fs = FakeFileSystem()
    fs.add_text("manual.txt", ALARM)
    app = _app(fs)
    app.add_path("manual.txt")

    info = app.remove("manual.txt")
    assert info["document_count"] == 0


def test_upload_writes_bytes_then_indexes() -> None:
    fs = FakeFileSystem()
    app = _app(fs)

    info = app.add_upload("parts.txt", b"PARTS\n\nThe drive belt is PN 44-19A.\n")

    assert info["document_count"] == 1
    assert any("uploads/parts.txt" in d["source_path"] for d in info["documents"])


@pytest.fixture
def base_url() -> Iterator[str]:
    fs = FakeFileSystem()
    fs.add_text("manual.txt", ALARM)
    app = _app(fs)
    app.add_path("manual.txt")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Handler, app=app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_http_serves_the_page(base_url: str) -> None:
    with urllib.request.urlopen(base_url + "/") as response:
        body = response.read().decode("utf-8")
    assert "<!DOCTYPE html>" in body
    assert "Cogwright" in body


def test_http_info_endpoint(base_url: str) -> None:
    with urllib.request.urlopen(base_url + "/api/info") as response:
        info = json.loads(response.read())
    assert info["document_count"] == 1


def test_http_ask_streams_server_sent_events(base_url: str) -> None:
    url = base_url + "/api/ask?q=How%20do%20I%20clear%20alarm%20204%3F"
    events = []
    with urllib.request.urlopen(url) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:") :].strip()))
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"]["found"] is True
