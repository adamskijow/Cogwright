<!--
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: 2026 The Cogwright Authors
-->

# Releasing

Cogwright is distributed on PyPI as `cogwright-rag` (the `cogwright` name was
already taken). The import package and the CLI command stay `cogwright`.

## One-time PyPI setup

Releases use [trusted publishing](https://docs.pypi.org/trusted-publishers/), so
no API token is stored in the repository. Once, on PyPI:

1. Register (or claim) the project `cogwright-rag`.
2. Add a trusted publisher pointing at this repository, the `release.yml`
   workflow, and the `pypi` environment.

## Cutting a release

1. Bump the version in both `pyproject.toml` (`project.version`) and
   `src/cogwright/__init__.py` (`__version__`) to the same value.
2. Make sure the checks pass locally:

   ```sh
   uv run ruff check . && uv run mypy && uv run pytest
   ```

3. Build and verify the distributions:

   ```sh
   uv build
   uvx twine check dist/*
   ```

4. Commit, then tag and push:

   ```sh
   git tag v0.1.0
   git push origin v0.1.0
   ```

Pushing the tag runs `release.yml`, which builds, checks, and publishes to PyPI.
A published version cannot be replaced, so verify the build before tagging.

## Verifying an install

```sh
pip install cogwright-rag
cogwright --version
```
