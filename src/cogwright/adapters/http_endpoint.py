# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Reference LLM and embedding clients for a configurable HTTP endpoint.

They speak the widely used JSON shape for chat completions and embeddings, over
the routes ``/v1/chat/completions`` and ``/v1/embeddings``, using only the
standard library. The tool therefore pulls in no model SDK and stays
offline-capable: the single network destination is the endpoint the user
configures. That endpoint can be a model server running on the same machine or a
remote one reached by base URL and key; no provider is hardwired. Connection
failures are surfaced as :class:`ModelUnavailableError` so the CLI can report
them cleanly.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from typing import Any

from ..core.errors import ModelUnavailableError
from ..core.models import Message, Vector


class _Endpoint:
    """Shared HTTP plumbing for the LLM and embedder clients."""

    def __init__(self, base_url: str, api_key: str | None, timeout: float) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def _url(self, route: str) -> str:
        return f"{self._base}/{route.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _request(self, route: str, payload: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            self._url(route),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

    def post_json(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._request(route, payload)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise ModelUnavailableError(self._describe(exc)) from exc
        parsed: dict[str, Any] = json.loads(body)
        return parsed

    def post_stream(self, route: str, payload: dict[str, Any]) -> Iterator[str]:
        request = self._request(route, payload)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    yield data
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise ModelUnavailableError(self._describe(exc)) from exc

    def reachable(self) -> bool:
        request = urllib.request.Request(self._url("models"), method="GET")
        for name, value in self._headers().items():
            request.add_header(name, value)
        try:
            urllib.request.urlopen(request, timeout=min(self._timeout, 5.0)).close()
            return True
        except urllib.error.HTTPError:
            # The server answered, even if the route is not implemented; it is up.
            return True
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def _describe(self, exc: Exception) -> str:
        return f"could not reach endpoint at {self._base}: {exc}"


class HttpLLMClient:
    """Chat-completions client behind the :class:`LLMClient` seam."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._endpoint = _Endpoint(base_url, api_key, timeout)
        self._model = model

    def complete(self, messages: Sequence[Message]) -> str:
        payload = self._payload(messages, stream=False)
        data = self._endpoint.post_json("chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailableError(f"unexpected completion response: {exc}") from exc
        return str(content)

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        payload = self._payload(messages, stream=True)
        for data in self._endpoint.post_stream("chat/completions", payload):
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content")
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
            if delta:
                yield str(delta)

    def available(self) -> bool:
        return self._endpoint.reachable()

    def _payload(self, messages: Sequence[Message], stream: bool) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "temperature": 0.0,
        }


class HttpEmbedder:
    """Embeddings client behind the :class:`Embedder` seam."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._endpoint = _Endpoint(base_url, api_key, timeout)
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        payload = {"model": self._model, "input": list(texts)}
        data = self._endpoint.post_json("embeddings", payload)
        try:
            items = sorted(data["data"], key=lambda item: item.get("index", 0))
            return [[float(x) for x in item["embedding"]] for item in items]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailableError(f"unexpected embedding response: {exc}") from exc

    def available(self) -> bool:
        return self._endpoint.reachable()
