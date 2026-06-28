<!--
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: 2026 The Cogwright Authors
-->

# Contributing to Cogwright

Thanks for your interest in improving Cogwright. This guide covers the local
workflow and the few conventions the project keeps.

## Setup

Cogwright targets Python 3.12 and uses [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Before opening a pull request

Run the same three checks that CI runs, and make sure they pass:

```sh
uv run ruff check .    # lint and import order
uv run mypy            # strict static type checking
uv run pytest          # tests, including the end-to-end fixture run
```

## Architecture

Keep the separation intact:

- `src/cogwright/core` is pure. It holds all retrieval and decision logic behind
  the protocol seams and must not import a concrete model, vector database, web
  framework, or anything that performs real input or output.
- `src/cogwright/adapters` and `src/cogwright/cli` are the only places real I/O
  lives. New parsers, stores, or model clients go here, behind the existing
  protocols, so the core never has to change to support them.

New behavior in the core should come with unit tests that use the fakes in
`tests/fakes.py`. No test may contact a live model or endpoint.

## Conventions

A couple of small rules are enforced automatically by `tests/test_repo_policy.py`:

- Every source file begins with an SPDX header
  (`SPDX-License-Identifier: MIT`).
- Do not use em dashes in code, comments, or docs.

Commit messages describe the change plainly and do not include tool or assistant
attribution.

## License

By contributing, you agree that your contributions are licensed under the MIT
License.
