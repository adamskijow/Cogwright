<!--
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: 2026 The Cogwright Authors
-->

# Cogwright

Cogwright is a local-first retrieval engine for technical equipment
documentation. You point it at a corpus of manuals, service bulletins, and parts
lists; a technician asks a question in plain language; and it returns a cited,
step-by-step answer grounded only in those documents, including alarm and stop
code lookup and references to the correct part.

It is built for field troubleshooting of industrial and mechanical equipment,
where the right answer is already written down somewhere across hundreds of pages
and the job is to find it, ground the response in it, and show the source.

## What it is, and what it is not

Cogwright **is** a knowledge-retrieval engine over a fixed set of documents:

- Ingest hard industrial documents well.
- Retrieve precisely, combining meaning-based search with exact identifier
  lookup.
- Answer with citations back to the source page, or say plainly when the answer
  is not in the documents.

Cogwright is **not**:

- an equipment-monitoring or live-telemetry platform,
- a multi-tenant or hosted service,
- an authoring or publishing suite for manuals,
- a general web search tool.

The scope is deliberately one slice, done well.

## Local-first and private by design

- **Offline-capable.** Nothing is required beyond your documents and a model
  endpoint you choose. There is no required cloud service.
- **No telemetry.** The tool collects and transmits nothing about you or your
  corpus.
- **One network destination.** The only outbound calls are to the single model
  and embedding endpoint you configure. That endpoint can run on the same
  machine, so a fully air-gapped deployment is possible.

Your documents stay where you put them, and the index is a plain file on your
disk.

## Supported documents

Out of the box, with no extra dependencies:

- plain text (`.txt`, `.text`),
- Markdown (`.md`, `.markdown`),
- born-digital PDFs (`.pdf`), including tables, where the page has a real text
  layer.

Scanned and photographed PDF pages (no text layer) are supported through an
optional optical-recognition engine. Install the extra and recognized text is
structured exactly like born-digital text:

```sh
uv sync --extra ocr
```

Recognition quality depends on the engine and the scan, so this path is newer
than the born-digital one. Exploded-diagram region understanding is still future
work; see the [roadmap](#roadmap). Both sit behind the same parser seam, so they
were added, and can be extended, without touching the core.

## Installation

Cogwright targets Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for the
environment and dependency management.

```sh
uv sync
```

This creates a virtual environment and installs the pinned dependencies from the
lockfile. The single third-party runtime dependency is a permissively licensed
PDF toolkit; everything else, including the model client, uses the standard
library.

## Pointing it at a model endpoint

Cogwright is model-agnostic. The reference client speaks the common JSON shape
for chat completions and embeddings over the routes `/v1/chat/completions` and
`/v1/embeddings`. Any local or remote model server that exposes that shape works;
no provider is hardwired.

Configure the endpoint with flags or environment variables:

| Setting          | Flag                  | Environment variable          | Default                     |
| ---------------- | --------------------- | ----------------------------- | --------------------------- |
| Base URL         | `--base-url`          | `COGWRIGHT_BASE_URL`          | `http://localhost:8000/v1`  |
| API key          | `--api-key`           | `COGWRIGHT_API_KEY`          | none                        |
| Chat model       | `--llm-model`         | `COGWRIGHT_LLM_MODEL`        | `local-chat-model`          |
| Embedding model  | `--embedding-model`   | `COGWRIGHT_EMBEDDING_MODEL`  | `local-embedding-model`     |
| Index path       | `--index`             | `COGWRIGHT_INDEX`            | `.cogwright/index.json`     |

The model names are placeholders; set them to whatever your endpoint serves. If
the endpoint cannot be reached, Cogwright reports the problem and exits with a
non-zero status rather than crashing.

## Usage

Build an index from a folder of documents:

```sh
uv run cogwright ingest ./manuals --base-url http://localhost:8000/v1 \
  --embedding-model your-embedding-model
```

Ask a question:

```sh
uv run cogwright ask "How do I clear alarm 204?" \
  --base-url http://localhost:8000/v1 \
  --llm-model your-chat-model
```

The answer streams as it is generated, followed by any referenced identifiers
and the source pages it was drawn from:

```
Follow these steps:
1. Stop the unit and allow the gearbox to cool for ten minutes.
2. Check the coolant level and refill to the cold mark if it is low.
3. Clear alarm 204 from the panel and restart the unit. [<id>]

Referenced identifiers: AL-204

Sources:
  - manuals/series7_conveyor_manual.txt (page 3, section: ALARM AND STOP CODE REFERENCE)
```

A bare alarm code, stop code, fault, error, diagnostic trouble code, warning, or
part number resolves to the exact passage that documents it. When the corpus does
not contain the answer, Cogwright says so instead of inventing a procedure.

To see which passages retrieval surfaced and how they ranked, add
`--show-retrieved` to `ask`.

## Evaluating retrieval quality

Retrieval quality can be measured against a graded dataset without calling the
model. A dataset pairs each question with the source pages it should surface, the
identifiers it should resolve, and whether it is answerable at all.

```sh
uv run cogwright eval tests/fixtures/sample_manual/eval.json \
  --base-url http://localhost:8000/v1 --embedding-model your-embedding-model
```

It reports found accuracy, page hit rate, code-resolution accuracy, and
not-found accuracy, each with its counts, so a change in chunking or retrieval
shows up as a number.

## Design

Cogwright is split into a pure core and a thin layer that wires real input and
output.

```
documents -> DocumentParser -> chunker -> Embedder ---> index (on disk)
                                  |                       |
                                  +--> code index --------+
                                                          |
question -> code detect + Embedder -> hybrid retrieval -> PromptBuilder -> LLMClient
                                                          |
                                                          +--> CitationMapper -> cited answer
```

### Strict separation behind protocol seams

The core library holds all retrieval and decision logic and depends only on five
protocols, never on a concrete model, vector store, or web framework:

- `FileSystem`, `DocumentParser`, `Embedder`, `LLMClient`, `VectorStore`.

The adapter layer supplies real implementations (disk access, text and PDF
parsers, the HTTP model client, and an in-memory vector store), and the test
suite supplies deterministic fakes. This keeps the core fully unit-testable with
no live model.

### Structure-aware ingestion

A naive splitter destroys exactly the content a technician needs whole. The
chunker respects document structure:

- a table becomes a single chunk with its rows preserved,
- a run of numbered procedure steps stays together and is never split,
- headings open a new chunk and travel with the content beneath them as section
  context.

Chunk identifiers are derived from their content, so re-ingesting an unchanged
document yields the same identifiers and citations stay stable across runs.

### Code and part lookup as a first-class index

A dedicated pass detects and indexes alarm codes, stop codes, fault and error
identifiers, diagnostic trouble codes, warning codes, and part numbers as exact
lookup keys. The rules are configurable patterns, not hardcoded schemes, so a
bare query such as `AL-204`, `alarm 204`, or `SC-12` resolves to the precise
passage that documents it.

### Hybrid retrieval and grounded answers

Retrieval merges two signals: semantic top-k from the embedding store, and exact
identifier lookup from the code index. Exact matches are precise, so they rank
above purely semantic hits, and a passage that wins on both is ranked highest.

The prompt instructs the model to answer only from the retrieved passages, to
give numbered steps for procedures, to surface the relevant identifiers, to cite
each passage it uses, and to say plainly when the answer is not present. If
retrieval finds nothing relevant, the model is never called and a clean
not-found result is returned, so there is no hallucinated procedure.

## Testing

The core is fully unit-tested with fakes for every seam, and the suite includes
an end-to-end run of ingest and ask against a small synthetic manual for an
invented machine.

```sh
uv run pytest          # run the test suite, including the end-to-end fixture run
uv run mypy            # strict static type checking
uv run ruff check .    # lint and import order
```

No live model is used in the tests. The same three checks run in continuous
integration on every push and pull request.

## Licensing

Cogwright is released under the [MIT License](LICENSE), and every source file
carries an SPDX header. All runtime dependencies are permissively licensed (MIT,
BSD, Apache-2.0, and HPND). Copyleft-licensed toolkits are deliberately avoided,
including in the PDF path, to keep the dependency tree MIT-compatible.

## Roadmap

Landed since the first milestone:

- Scanned-page optical recognition behind the parser seam, available through the
  optional `ocr` extra.
- A retrieval evaluation harness and `eval` command.

Still ahead. The seams needed for these are already in place.

- Layout-aware recognition and quality tuning for low-quality scans.
- Region understanding for exploded diagrams and callouts.
- Video ingestion.
- Live-telemetry and equipment-monitoring integration.
- Role-based access and multi-tenant accounts.
- A web interface.
- Any marketplace or publishing pipeline for documents.

## Configuration defaults and assumptions

Where the design left a choice open, the following defaults were taken and can be
revisited:

- The model and embedding backends are reached through one HTTP endpoint that
  follows the common chat-completions and embeddings JSON shape.
- PDF parsing targets born-digital files with a recoverable text layer.
- Heading detection in plain text treats all-caps lines and Markdown headings as
  section titles, which matches how equipment manuals are commonly laid out.
