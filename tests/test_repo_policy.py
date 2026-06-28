# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 The Cogwright Authors
"""Repository policy guards that run as part of the normal test suite.

These keep two project conventions from regressing: every source file carries an
SPDX license header, and no source or documentation file contains an em dash.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Directories that hold generated output or third-party code, not ours to police.
_SKIP_DIRS = {
    ".git",
    ".venv",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "dist",
    "build",
    ".cogwright",
}

# Text we own and check for the em-dash convention.
_TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".yml", ".yaml", ".txt", ".cfg"}

# Written as an escape so this guard file does not itself trip the check.
_EM_DASH = "\u2014"


def _owned_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def test_every_python_source_file_has_an_spdx_header() -> None:
    offenders: list[str] = []
    for path in _owned_files():
        if path.suffix != ".py":
            continue
        head = path.read_text(encoding="utf-8").splitlines()[:5]
        if not any("SPDX-License-Identifier: MIT" in line for line in head):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"missing SPDX header: {offenders}"


def test_no_owned_text_file_contains_an_em_dash() -> None:
    offenders: list[str] = []
    for path in _owned_files():
        if path.suffix not in _TEXT_SUFFIXES:
            continue
        if _EM_DASH in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"em dash found in: {offenders}"
