# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""A reference DiagramAnalyzer backed by a multimodal chat endpoint.

It sends a rendered page image to an OpenAI-compatible multimodal chat model and
asks it to transcribe the callouts, labels, and captions printed on the diagram,
one per line. It uses only the standard library: the image is inlined as a base64
data URL in the chat request, so no extra dependency is needed and the single
network destination remains the configured endpoint.
"""

from __future__ import annotations

import base64

from ..core.errors import ModelUnavailableError
from .http_endpoint import HttpEndpoint

_PROMPT = (
    "This image is a technical diagram or an exploded parts illustration from "
    "equipment documentation. Transcribe every callout, label, and caption that "
    "is printed on it, exactly as written, one per line. Include any part numbers "
    "or identifiers. Do not describe the image or add commentary. If there is no "
    "readable text, reply with the single word NONE."
)


class VisionDiagramAnalyzer:
    """Transcribes diagram callouts using a multimodal chat model."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        prompt: str = _PROMPT,
        max_tokens: int = 512,
    ) -> None:
        self._endpoint = HttpEndpoint(base_url, api_key, timeout)
        self._model = model
        self._prompt = prompt
        # Bound the reply so a model told to "transcribe everything" cannot run on
        # indefinitely, which otherwise stalls a whole ingest on one busy figure.
        self._max_tokens = max_tokens

    def describe(self, image: bytes) -> list[str]:
        data_url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        data = self._endpoint.post_json("chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailableError(f"unexpected vision response: {exc}") from exc
        return _clean_lines(str(content))

    def available(self) -> bool:
        return self._endpoint.reachable()


def _clean_lines(content: str) -> list[str]:
    """Split a transcription into callout strings, dropping list markers and noise."""

    lines: list[str] = []
    for raw in content.splitlines():
        line = raw.strip().lstrip("-*").strip()
        # Drop a leading enumerator like "1." or "2)" the model may have added.
        if line[:1].isdigit() and line[1:2] in {".", ")"}:
            line = line[2:].strip()
        if not line or line.upper() == "NONE":
            continue
        lines.append(line)
    return lines
