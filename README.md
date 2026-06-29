<!--
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: 2026 The Cogwright Authors
-->

<p align="center">
  <img src="https://raw.githubusercontent.com/adamskijow/Cogwright/main/assets/cogwright_banner.gif" alt="Cogwright: grounded, cited answers from your documents" width="820">
</p>

[![PyPI](https://img.shields.io/pypi/v/cogwright-rag)](https://pypi.org/project/cogwright-rag/)
[![CI](https://github.com/adamskijow/Cogwright/actions/workflows/ci.yml/badge.svg)](https://github.com/adamskijow/Cogwright/actions/workflows/ci.yml)

Cogwright answers questions about your technical documentation. Point it at a
folder of manuals, specs, runbooks, or service bulletins, ask in plain language,
and get a step-by-step answer grounded only in those documents, with citations to
the source page. Bare identifiers resolve too: an alarm code, an error code, or a
part number jumps straight to the passage that defines it.

It runs locally against an OpenAI-compatible model endpoint you choose, so your
documents and the index stay on your machine and the only network calls go to
that one endpoint.

Install it, point `--base-url` at your model server (local or hosted), and set
the model names to whatever it serves:

```sh
pip install cogwright-rag

cogwright ingest ./manuals \
  --base-url http://localhost:8000/v1 --embedding-model nomic-embed-text

cogwright ask "How do I clear alarm 204?" \
  --base-url http://localhost:8000/v1 \
  --llm-model llama3.2:1b --embedding-model nomic-embed-text --min-score 0.55
```

```
To clear alarm 204, follow these steps:
1. Stop the unit and allow the gearbox to cool for ten minutes.
2. Check the coolant level and refill to the cold mark if it is low.
3. Clear alarm 204 from the panel and restart the unit.

Referenced identifiers: AL-204

Sources:
  - manuals/series7_conveyor_manual.txt (page 3, section: ALARM AND STOP CODE REFERENCE)
```

When the corpus does not contain the answer, Cogwright says so instead of
inventing one. Prefer a browser? `cogwright serve` opens the same answers in a
local web interface, with a corpus manager for adding and removing documents.

## How it works

**Ingest** parses each document into structure-aware chunks, where a table stays
whole and a numbered procedure stays together, embeds them, and writes a single
JSON index to disk. A dedicated pass indexes every alarm, stop, fault, error,
diagnostic, warning, and part identifier as an exact lookup key.

**Ask** embeds the question and retrieves by two signals at once: semantic
similarity and exact identifier lookup, with exact matches ranked above fuzzy
ones. Those passages, and only those, go to the model with instructions to answer
from them, number any steps, surface the identifiers, and cite each passage. If
nothing clears the relevance bar, the model is never called.

```
ingest:  documents -> parse -> chunk -> embed ------+
                          \--> identifier index ----+--> index.json

ask:     question -> embed + detect identifiers -> hybrid retrieval
                  -> prompt (retrieved context only) -> model -> cited answer
```

## Install

Cogwright targets Python 3.12. It is published on PyPI as `cogwright-rag` (the
import package and the `cogwright` command keep the shorter name):

```sh
pip install cogwright-rag          # core install
pip install "cogwright-rag[ocr]"   # add scanned-page recognition
```

For development, use [uv](https://docs.astral.sh/uv/):

```sh
uv sync                  # core install
uv sync --extra ocr      # add scanned-page recognition
```

The only third-party runtime dependency is a PDF toolkit. The model client,
vector math, and CLI are all standard library.

## Configure the endpoint

Cogwright talks to any OpenAI-compatible endpoint, using the routes
`/v1/chat/completions` and `/v1/embeddings`. That can be a model server on the
same machine or a hosted API; the implementation is not tied to one provider, and
it has been validated against a local server running a small chat model and an
embedding model. Configure it with flags or environment variables:

| Setting          | Flag                | Environment variable          | Default                    |
| ---------------- | ------------------- | ----------------------------- | -------------------------- |
| Base URL         | `--base-url`        | `COGWRIGHT_BASE_URL`         | `http://localhost:8000/v1` |
| API key          | `--api-key`         | `COGWRIGHT_API_KEY`          | none                       |
| Chat model       | `--llm-model`       | `COGWRIGHT_LLM_MODEL`        | `local-chat-model`         |
| Embedding model  | `--embedding-model` | `COGWRIGHT_EMBEDDING_MODEL`  | `local-embedding-model`    |
| Vision model     | `--vision-model`    | `COGWRIGHT_VISION_MODEL`     | `local-vision-model`       |
| Index path       | `--index`           | `COGWRIGHT_INDEX`            | `.cogwright/index.json`    |

The model names are placeholders; set them to whatever your endpoint serves. An
unreachable endpoint is reported with a non-zero exit, never a crash.

## Commands

### ingest

```sh
cogwright ingest <paths...> [--ocr] [--diagrams]
```

Paths are files or folders. `--ocr` recognizes scanned PDF pages and needs the
`ocr` extra. `--diagrams` transcribes figure callouts with a multimodal model set
by `--vision-model`. `ingest` builds a fresh index, recording which documents it
holds, when it was built, and the embedding model that produced the vectors.

### update, remove, info

Maintain an index without rebuilding it from scratch:

```sh
cogwright update <paths...>     # add new documents, refresh changed ones, skip unchanged
cogwright remove <paths...>     # drop documents, matched by path or file name
cogwright info                  # show the documents, counts, model, and timestamps
```

`update` compares a content hash per document, so re-running it only re-embeds
what actually changed. Updating with a different embedding model than the index
was built with is refused, and a query run with a mismatched model warns, because
vectors from different models are not comparable.

### ask

```sh
cogwright ask "<question>" [--top-k N] [--min-score S] [--no-stream] \
  [--show-retrieved] [--json]
```

The answer streams as it is generated. `--show-retrieved` prints the ranked
passages and their scores first, which is how you see what retrieval is doing.
`--json` asks the model for a structured reply (steps as a list, the passages it
used named explicitly), which gives reliable numbered steps and precise
citations with a capable model and falls back to the prose path otherwise.

### eval

```sh
cogwright eval <dataset.json> [--min-score S]
```

Scores retrieval against a graded dataset without calling the chat model. See
[calibrating relevance](#calibrating-relevance).

### serve

```sh
cogwright serve [--host 127.0.0.1] [--port 8765]
```

Runs a local web interface in the browser: a search box with streaming cited
answers and a retrieval inspector, plus a corpus view for adding documents (by
path or drag-and-drop) and removing them. It is a standard-library server with a
bundled offline page, so it pulls in no web framework and serves only on the host
you bind. Your documents and the index never leave the machine.

## Documents it understands

- **Text and Markdown** (`.txt`, `.text`, `.md`, `.markdown`). Headings, numbered
  steps, and pipe tables are recovered.
- **Born-digital PDFs** with a real text layer, including tables, which are lifted
  out as structured blocks. Real page numbers are kept for citations.
- **Scanned PDF pages**, with the `ocr` extra. A page with little text and a
  dominant image is rendered and recognized, then structured like any other page.
- **Diagram callouts**, with `--diagrams`. A figure is sent to a vision model and
  the printed labels and part numbers become searchable.

## Identifier lookup

A query that is a bare code resolves to the exact passage that documents it. The
built-in patterns detect and normalize:

| You type                          | Resolves to |
| --------------------------------- | ----------- |
| `alarm 204`, `AL-204`, `AL204`    | `AL-204`    |
| `STOP CODE 12`, `SC-12`           | `SC-12`     |
| `fault 09`                        | `F-09`      |
| `error 30`                        | `E-30`      |
| `DTC P0420`                       | `DTC-P0420` |
| `warning 18`                      | `W-18`      |
| `PN 44-19A`, `P/N 44-19A`         | `PN-44-19A` |

The patterns are configuration rather than hardcoded, so a deployment can add its
own identifier schemes.

## Calibrating relevance

The not-found decision rests on one cosine cutoff, `--min-score`. It is
embedding-model dependent: different models place related and unrelated text at
different similarity ranges, so there is no universal value (the default, 0.45,
suits typical normalized models). Calibrate it with `eval`. A dataset pairs each
question with the pages it should surface, the identifiers it should resolve, and
whether it is answerable at all:

```json
{ "question": "How do I clear alarm 204?", "expected_pages": [3],
  "expected_codes": ["AL-204"], "should_find": true }
```

Raise `--min-score` until the unanswerable cases report not-found while the real
ones still resolve, then pass that value to both `eval` and `ask`. The harness
reports found accuracy, page hit rate, code-resolution accuracy, and not-found
accuracy, each with its counts.

## Architecture

A pure core holds all retrieval and decision logic; a thin adapter and CLI layer
does the real input and output. The core depends only on protocols, never on a
concrete model, store, or framework:

- core seams: `FileSystem`, `DocumentParser`, `Embedder`, `LLMClient`,
  `VectorStore`
- ingestion seams: `OcrEngine`, `DiagramAnalyzer`

Adapters supply the real implementations (disk, text and PDF parsers, the HTTP
client, an in-memory cosine store, an OCR engine, a vision analyzer); the tests
supply fakes. To add a vector database, a different model API, or a new document
type, implement the seam and the core does not change.

## Testing

```sh
uv run pytest          # tests, including end-to-end ingest and ask on a sample manual
uv run mypy            # strict type checking
uv run ruff check .    # lint
```

Every seam has a deterministic fake, so the suite needs no live model. Tests for
the real OCR engine run only where it is installed and skip otherwise. All three
checks run in CI on every push and pull request, alongside a guard that fails on a
missing license header.

## Privacy

No telemetry and no required cloud service. The only outbound traffic is to the
endpoint you configure, which can run on the same machine, so a fully air-gapped
deployment is possible. Documents stay where you put them, and the index is a
plain file you control.

## License

MIT, with an SPDX header on every source file. Every dependency is permissively
licensed; copyleft toolkits are avoided, including in the PDF path, to keep the
tree MIT-compatible. See [CONTRIBUTING.md](CONTRIBUTING.md) to work on it.

## Status

Released and validated against a live local model: text and born-digital PDF
ingestion, hybrid retrieval with identifier lookup, grounded cited answers, the
structured-JSON answer mode, incremental index updates, the evaluation harness,
and the local web interface. Scanned-page OCR and diagram transcription are
available behind their seams. Region-level diagram cropping, tuning for
low-quality scans, and more corpus formats are future work, each fitting an
existing seam.
