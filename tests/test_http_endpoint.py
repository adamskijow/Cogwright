# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Integration tests for the HTTP model and embedding adapters.

A throwaway in-process HTTP server stands in for the endpoint, so the real
urllib client, the server-sent-events streaming parser, and the response shapes
are all exercised without depending on any external service.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cogwright.adapters.http_endpoint import HttpEmbedder, HttpLLMClient
from cogwright.core.errors import ModelUnavailableError
from cogwright.core.models import Message

_STREAM_PIECES = ["Follow ", "these ", "steps."]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # silence test noise
        return

    def do_GET(self) -> None:
        if self.path.endswith("/models"):
            self._json(200, {"data": [{"id": "test-model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/embeddings"):
            self._embeddings(payload)
        elif self.path.endswith("/chat/completions"):
            self._completions(payload)
        else:
            self._json(404, {"error": "not found"})

    def _embeddings(self, payload: dict[str, object]) -> None:
        inputs = payload.get("input", [])
        assert isinstance(inputs, list)
        data = [
            {"index": i, "embedding": [float(len(str(text))), float(i)]}
            for i, text in enumerate(inputs)
        ]
        self._json(200, {"data": data})

    def _completions(self, payload: dict[str, object]) -> None:
        if payload.get("stream"):
            self._stream()
            return
        self._json(
            200,
            {"choices": [{"message": {"role": "assistant", "content": "Full answer."}}]},
        )

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in _STREAM_PIECES:
            chunk = {"choices": [{"delta": {"content": piece}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _json(self, status: int, body: dict[str, object]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@pytest.fixture
def base_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_embedder_returns_one_vector_per_input(base_url: str) -> None:
    embedder = HttpEmbedder(base_url, model="test-embed")
    vectors = embedder.embed(["abc", "de"])
    assert vectors == [[3.0, 0.0], [2.0, 1.0]]


def test_embedder_empty_input_makes_no_call(base_url: str) -> None:
    assert HttpEmbedder(base_url, model="test-embed").embed([]) == []


def test_llm_complete_parses_message_content(base_url: str) -> None:
    llm = HttpLLMClient(base_url, model="test-llm")
    assert llm.complete([Message(role="user", content="hi")]) == "Full answer."


def test_llm_stream_yields_delta_pieces(base_url: str) -> None:
    llm = HttpLLMClient(base_url, model="test-llm")
    pieces = list(llm.stream([Message(role="user", content="hi")]))
    assert pieces == _STREAM_PIECES
    assert "".join(pieces) == "Follow these steps."


def test_available_is_true_for_live_server_and_false_when_down(base_url: str) -> None:
    assert HttpLLMClient(base_url, model="test-llm").available() is True
    down = HttpLLMClient("http://127.0.0.1:9/v1", model="test-llm")
    assert down.available() is False


def test_unreachable_endpoint_raises_model_unavailable() -> None:
    embedder = HttpEmbedder("http://127.0.0.1:9/v1", model="x", timeout=2.0)
    with pytest.raises(ModelUnavailableError):
        embedder.embed(["anything"])
