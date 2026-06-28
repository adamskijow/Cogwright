# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""A real, disk-backed implementation of the FileSystem seam."""

from __future__ import annotations

import os
from collections.abc import Sequence


class RealFileSystem:
    """Reads and writes the local disk. The only filesystem the CLI wires in."""

    def read_bytes(self, path: str) -> bytes:
        with open(path, "rb") as handle:
            return handle.read()

    def read_text(self, path: str) -> str:
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def write_text(self, path: str, data: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(data)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def list_files(self, path: str, extensions: Sequence[str]) -> list[str]:
        if os.path.isfile(path):
            return [path]
        lowered = tuple(ext.lower() for ext in extensions)
        found: list[str] = []
        for root, _dirs, files in os.walk(path):
            for name in files:
                if not lowered or name.lower().endswith(lowered):
                    found.append(os.path.join(root, name))
        return sorted(found)
