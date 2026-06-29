# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""A local, offline web interface for asking questions and managing the corpus.

The browser talks to a small standard-library server running on the same
machine, so the documents never leave the computer and no web framework is
pulled in. The pure core is unchanged; this is another app layer beside the CLI.
"""

from __future__ import annotations

from .app import WebApp
from .server import serve

__all__ = ["WebApp", "serve"]
