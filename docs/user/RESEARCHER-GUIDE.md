# Researcher Guide

Use the current contracts as the reproducibility boundary:

- [Definition](definition.md), [Workload](workload.md), and [Solution](solution.md)
  describe accepted benchmark inputs.
- [Trace](trace.md) is the canonical evaluation output.
- [SOLAR Boundary](../SOLAR-BOUNDARY.md) defines formal bound ownership.
- [Scoring](../SCORING-V3.md) defines the formula and unavailable official
  authority state.
- [Evaluator Contract](EVALUATOR-CONTRACT.md) exposes the same boundary in a
  machine-readable form.

Record the exact corpus manifest, source revision, solution content hash,
architecture identity, ROCm environment, and Trace JSONL for each experiment.
Treat profile summaries, static evidence, environment snapshots, feedback, and
decision sidecars as diagnostic aids only.

Public schema parsers are strict: unknown fields, superseded field names, and
unsupported version identifiers are errors. The pinned upstream corpus importer
is the only layer that translates reviewed upstream field names into the current
Definition and Workload shapes.

Run `uv run sol-execbench contract --format json` to inspect the current
machine-readable evaluator boundary and `uv run pytest tests/` for the complete
test suite. ROCm hardware tests retain their explicit markers and skip when the
required device or toolchain is unavailable.
