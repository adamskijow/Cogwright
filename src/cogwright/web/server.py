# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""A small standard-library HTTP server for the web app.

It serves the bundled page, exposes a few JSON endpoints, and streams answers as
server-sent events. There is no framework: routing is a handful of path checks,
which keeps the dependency footprint at zero. It binds to the local host by
default so the interface stays on the machine it runs on.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..core.errors import CogwrightError, ModelUnavailableError
from .app import WebApp


class _Handler(BaseHTTPRequestHandler):
    server_version = "Cogwright"

    def __init__(self, *args: Any, app: WebApp, **kwargs: Any) -> None:
        self._app = app
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # quieter logging
        return

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if route.path in ("/", "/index.html"):
            self._html(self._app.page())
        elif route.path == "/api/info":
            self._guarded(lambda: self._json(200, self._app.info()))
        elif route.path == "/api/ask":
            self._ask(parse_qs(route.query))
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        route = urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if route.path == "/api/ingest":
            payload = _load_json(body)
            self._guarded(lambda: self._json(200, self._app.add_path(payload["path"])))
        elif route.path == "/api/remove":
            payload = _load_json(body)
            self._guarded(
                lambda: self._json(200, self._app.remove(payload["source_path"]))
            )
        elif route.path == "/api/upload":
            name = (parse_qs(urlparse(self.path).query).get("filename") or ["upload"])[0]
            self._guarded(lambda: self._json(200, self._app.add_upload(name, body)))
        elif route.path == "/api/settings":
            payload = _load_json(body)
            self._guarded(
                lambda: self._json(
                    200,
                    self._app.update_settings(
                        llm_model=(payload.get("llm_model") or None),
                        min_score=_opt_float(payload.get("min_score")),
                    ),
                )
            )
        else:
            self._json(404, {"error": "not found"})

    def _ask(self, params: dict[str, list[str]]) -> None:
        question = (params.get("q") or [""])[0].strip()
        if not question:
            self._json(400, {"error": "missing question"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for event in self._app.ask_stream(question):
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _guarded(self, action: Callable[[], None]) -> None:
        try:
            action()
        except ModelUnavailableError as exc:
            self._json(502, {"error": str(exc)})
        except CogwrightError as exc:
            self._json(400, {"error": str(exc)})
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            self._json(400, {"error": "malformed request"})

    def _json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _html(self, markup: str) -> None:
        encoded = markup.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _load_json(body: bytes) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(body or b"{}")
    return parsed


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def serve(app: WebApp, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the server until interrupted."""

    server = ThreadingHTTPServer((host, port), partial(_Handler, app=app))
    print(f"Cogwright is serving at http://{host}:{port}  (press Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
