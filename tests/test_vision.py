# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Integration tests for the multimodal diagram analyzer.

An in-process HTTP server stands in for the endpoint, so the request shape (an
inlined base64 image) and the response parsing are exercised against the real
urllib client without depending on a vision model.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cogwright.adapters.vision import VisionDiagramAnalyzer
from cogwright.core.errors import ModelUnavailableError

_REPLY = "Figure 1: drive belt PN 44-19A\n- tensioner pulley\n2. idler arm\nNONE\n"
_captured: dict[str, object] = {}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        _captured["payload"] = payload
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": _REPLY}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def base_url() -> Iterator[str]:
    _captured.clear()
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


def test_describe_parses_callouts_and_drops_markers(base_url: str) -> None:
    analyzer = VisionDiagramAnalyzer(base_url, model="vision")
    captions = analyzer.describe(b"\x89PNG\r\n\x1a\n fake-image-bytes")

    # List markers and enumerators are stripped, and NONE is dropped.
    assert captions == [
        "Figure 1: drive belt PN 44-19A",
        "tensioner pulley",
        "idler arm",
    ]


def test_request_inlines_the_image_as_a_data_url(base_url: str) -> None:
    VisionDiagramAnalyzer(base_url, model="vision").describe(b"img-bytes")

    payload = _captured["payload"]
    assert isinstance(payload, dict)
    content = payload["messages"][0]["content"]
    kinds = {part["type"] for part in content}
    assert kinds == {"text", "image_url"}
    image_part = next(p for p in content if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_unreachable_endpoint_raises_model_unavailable() -> None:
    analyzer = VisionDiagramAnalyzer("http://127.0.0.1:9/v1", model="vision", timeout=2.0)
    with pytest.raises(ModelUnavailableError):
        analyzer.describe(b"img")
