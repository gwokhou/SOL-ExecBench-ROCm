# v1.19 Evidence Guide

This guide is the central researcher-facing entry point for v1.19 evidence
surfaces. It explains how to generate and interpret the sidecars and reports
added across Phases 83-87 without changing benchmark authority.

The v1.19 artifacts are sidecars/reports only. The canonical Trace, Definition,
Workload, Solution contracts are unchanged. Put another way: canonical Trace, Definition, Workload, Solution, correctness, timing, score, and evaluator contract semantics are unchanged.

## Claim Boundaries

Use the phrases below when describing v1.19 results:

- no full 235-problem paper validation
- no upstream SOLAR parity
- no score authority
- no leaderboard readiness
- no CDNA3-family validation, including MI300X, and no CDNA4 validation
- no native-host ROCm Matrix validation
- no new-hardware validation

These boundaries apply even when an artifact is complete, deterministic, and
well formed. A report can explain evidence coverage, gaps, and diagnostic risk;
it cannot upgrade a benchmark score, paper parity claim, leaderboard readiness
claim, or hardware-validation claim.

## Evidence Surfaces

| Surface | Primary command surface | Artifact role |
| --- | --- | --- |
| execution closure | `scripts/run_dataset.py` | Records scoped problem closure statuses and bounded execution provenance. |
| paper denominator report | `scripts/report_paper_denominator.py` | Accounts for denominator coverage, evidence gaps, and claim boundaries. |
| Matrix schema export | `scripts/export_matrix_schema.py` | Exports strict JSON Schemas for Matrix report contracts. |
| Matrix semantic diff | `scripts/diff_matrix_reports.py` | Compares Matrix reports for diagnostic compatibility drift. |
| AMD bound sanity | `scripts/report_amd_bound_sanity.py` | Checks existing AMD/SOL/SOLAR evidence consistency and bounded risk. |

All examples below use demo paths such as `out/v1_19_demo/...`. They are not
real hardware validation outputs and must not be reported as performance
results.

## Execution Closure

Execution closure is written by the dataset runner path when closure reporting
is enabled. It records one closure status per scoped problem, along with
relative trace/log refs and concise bounded notes.

Example command shape:

```bash
UV_CACHE_DIR=out/v1_19_demo/uv-cache uv run scripts/run_dataset.py \
  data/SOL-ExecBench/benchmark \
  --limit 5 \
  --output out/v1_19_demo/run-dataset \
  --execution-closure out/v1_19_demo/execution_closure.json
```

Interpretation:

- `attempted_passed` means the scoped problem was attempted and produced
  passing trace evidence for the configured run.
- `attempted_failed` means the problem was attempted and did not pass.
- `skipped_existing_pass` is valid only when matching prior trace and closure
  provenance authorize reuse.
- `missing_trace`, `derived_evidence_missing`, `filtered`, and other non-pass
  statuses are denominator facts, not benchmark success.

What it cannot prove:

- no full 235-problem paper validation by this sidecar alone; a paper-validation
  claim requires a separately reviewed complete evidence bundle as described in
  `docs/CLAIMS.md`
- no score authority
- no leaderboard readiness
- no new-hardware validation

## Paper Denominator Report

The paper denominator report combines explicit sidecar inputs into
`sol_execbench.paper_denominator_report.v1` JSON and Markdown. It is the safest
place to explain which paper-denominator records have evidence and which remain
missing, deferred, filtered, or unavailable.

Example command shape:

```bash
UV_CACHE_DIR=out/v1_19_demo/uv-cache uv run scripts/report_paper_denominator.py \
  --manifest out/v1_19_demo/manifest.json \
  --inventory out/v1_19_demo/inventory.json \
  --readiness out/v1_19_demo/readiness.json \
  --execution-closure out/v1_19_demo/execution_closure.json \
  --amd-score-report out/v1_19_demo/amd_score.json \
  --json-output out/v1_19_demo/paper_denominator.json \
  --markdown-output out/v1_19_demo/paper_denominator.md
```

Interpretation:

- Source refs and checksums identify which bounded inputs were used.
- Denominator rollups distinguish evidence present, evidence missing, deferred,
  unsupported, and unavailable states.
- Claim-boundary fields remain false for paper parity, leaderboard readiness,
  score authority, upstream SOLAR parity, and hardware validation upgrades.

What it cannot prove:

- no full 235-problem paper validation by this report alone; a paper-validation
  claim requires a separately reviewed complete evidence bundle as described in
  `docs/CLAIMS.md`
- no upstream SOLAR parity
- no native-host ROCm Matrix validation
- no CDNA3-family validation, including MI300X, and no CDNA4 validation

## Matrix Schema Export

Matrix schema export writes strict JSON Schema documents for `MatrixEntry` and
`RocmCompatibilityMatrixReport`. It helps researchers verify report shape
without running Docker, probing hardware, or changing benchmark semantics.

Example command shape:

```bash
UV_CACHE_DIR=out/v1_19_demo/uv-cache uv run scripts/export_matrix_schema.py \
  --model all \
  --output-dir out/v1_19_demo/matrix-schema
```

Interpretation:

- The exported schema identifies the Matrix contract version and object shape.
- Strict extra-field behavior documents the sidecar schema boundary.
- Schema export is documentation and validation support for report shape only.

What it cannot prove:

- no native-host ROCm Matrix validation
- no new-hardware validation
- no score authority
- no leaderboard readiness

## Matrix Semantic Diff

Matrix semantic diff compares two existing Matrix reports and emits diagnostic
JSON/Markdown describing status, severity, and semantic field changes.

Example command shape:

```bash
UV_CACHE_DIR=out/v1_19_demo/uv-cache uv run scripts/diff_matrix_reports.py \
  out/v1_19_demo/matrix.before.json \
  out/v1_19_demo/matrix.after.json \
  --json-out out/v1_19_demo/matrix_diff.json \
  --markdown-out out/v1_19_demo/matrix_diff.md
```

Interpretation:

- Diff entries are matched by target identity and validation scope.
- Severity explains compatibility drift risk for diagnostics and review.
- Diagnostic-only claim boundaries remain false for score, paper parity,
  leaderboard, and native-host authority.

What it cannot prove:

- no native-host ROCm Matrix validation
- no full 235-problem paper validation
- no leaderboard readiness
- no new-hardware validation

## AMD Bound Sanity

AMD bound sanity consumes explicit existing evidence paths and produces
`sol_execbench.amd_bound_sanity.v1` JSON/Markdown. It checks whether AMD SOL,
SOLAR-derived, closure, Matrix, and score artifacts are
internally consistent enough to discuss as bounded diagnostic evidence.

Example command shape:

```bash
UV_CACHE_DIR=out/v1_19_demo/uv-cache uv run scripts/report_amd_bound_sanity.py \
  --trace out/v1_19_demo/traces/demo.trace.jsonl \
  --execution-closure out/v1_19_demo/execution_closure.json \
  --amd-sol-artifact out/v1_19_demo/amd-sol/demo.amd-sol-v2.json \
  --solar-artifact out/v1_19_demo/solar/demo.solar-derivation.json \
  --compatibility-matrix out/v1_19_demo/matrix.json \
  --amd-score-report out/v1_19_demo/amd_score.json \
  --json-output out/v1_19_demo/amd_bound_sanity.json \
  --markdown-output out/v1_19_demo/amd_bound_sanity.md
```

Interpretation:

- Source refs and checksums show which existing artifacts were inspected.
- Statuses and warnings identify missing evidence, degraded estimates,
  provisional risk, or internal inconsistencies.
- Authority fields remain false for leaderboard, score upgrades, native-host
  Matrix validation, upstream SOLAR parity, and new hardware validation.

What it cannot prove:

- no upstream SOLAR parity
- no score authority
- no leaderboard readiness
- no CDNA3-family validation, including MI300X, and no CDNA4 validation
- no new-hardware validation

## CPU-Safe Verification

Use focused docs and contract guardrails for v1.19 wording:

```bash
UV_CACHE_DIR=out/v1_19_demo/uv-cache uv run pytest \
  tests/sol_execbench/test_research_release_docs.py \
  tests/sol_execbench/test_rocm_matrix_docs.py \
  tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_paper_denominator_fields_remain_sidecar_only \
  tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_amd_bound_sanity_fields_remain_sidecar_only -q

UV_CACHE_DIR=out/v1_19_demo/uv-cache uv run ruff check \
  tests/sol_execbench/test_research_release_docs.py
```

These checks are CPU-safe documentation and public-contract checks. They do not
run GPU probes, ROCm live validation, Docker builds, Docker containers,
dependency relocking, or hardware-marker tests.
