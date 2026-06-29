<!--
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: 2026 The Cogwright Authors
-->

# Example corpus

A tiny synthetic manual for an invented machine (the Series 7 Conveyor Unit) and
a graded evaluation dataset, so you can try Cogwright end to end in a minute.

The manual has an alarm and stop code reference, a startup procedure, and a parts
table, so it exercises the things Cogwright is built for: structure-aware
chunking, identifier lookup, and grounded cited answers.

## Try it

Point Cogwright at any OpenAI-compatible endpoint (a local model server works).
Set the model names to whatever your endpoint serves.

```sh
cogwright ingest examples/manuals \
  --base-url http://localhost:8000/v1 --embedding-model <embed-model>

cogwright ask "How do I clear alarm 204?" \
  --base-url http://localhost:8000/v1 \
  --llm-model <chat-model> --embedding-model <embed-model>
```

A bare code resolves directly:

```sh
cogwright ask "AL-204" --base-url http://localhost:8000/v1 \
  --llm-model <chat-model> --embedding-model <embed-model>
```

## Measure and calibrate

`eval.json` grades retrieval against the manual without calling the chat model.
Use it to calibrate `--min-score` for your embedding model:

```sh
cogwright eval examples/eval.json \
  --base-url http://localhost:8000/v1 --embedding-model <embed-model> --min-score 0.5
```
