# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Typed errors raised by the core and surfaced cleanly by the CLI."""

from __future__ import annotations


class CogwrightError(Exception):
    """Base class for all errors this package raises on purpose."""


class UnsupportedDocumentError(CogwrightError):
    """No registered parser handles the given file."""


class ModelUnavailableError(CogwrightError):
    """The configured LLM or embedding endpoint could not be reached."""
