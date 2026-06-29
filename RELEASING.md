<!--
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: 2026 The Cogwright Authors
-->

# Releasing

Cogwright is distributed on PyPI as
[`cogwright-rag`](https://pypi.org/project/cogwright-rag/) (the `cogwright` name
was already taken). The import package and the CLI command stay `cogwright`.
Version `0.1.0` was published, so the project already exists on PyPI.

## One-time PyPI setup

Releases use [trusted publishing](https://docs.pypi.org/trusted-publishers/), so
no API token is stored in the repository. Once, on the existing `cogwright-rag`
project on PyPI, add a trusted publisher pointing at this repository, the
`release.yml` workflow, and the `pypi` environment.

If you would rather use a token, add it as a repository secret named
`PYPI_API_TOKEN` (a project-scoped token is enough now that the project exists)
and set the publish step's `password` to `${{ secrets.PYPI_API_TOKEN }}`.

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

4. Commit, then tag and push (match the tag to the version):

   ```sh
   git tag v0.2.0
   git push origin v0.2.0
   ```

Pushing the tag runs `release.yml`, which builds, checks, creates a GitHub
Release with the built artifacts, and publishes to PyPI. A version already on
PyPI is skipped rather than failed, so the workflow is safe to re-run and a
version published by hand (with `uv publish`) still gets a clean GitHub Release
when its tag is pushed. A published version cannot be replaced, so verify the
build before tagging.

## Verifying an install

```sh
pip install cogwright-rag
cogwright --version
```
