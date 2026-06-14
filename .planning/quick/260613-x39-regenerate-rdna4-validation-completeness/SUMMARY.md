---
quick_id: 260613-x39
slug: regenerate-rdna4-validation-completeness
status: complete
---

# Quick Summary: Regenerate RDNA4 Validation Completeness Report

Regenerated the RDNA4 validation completeness artifacts from the internal RDNA4
report scripts.

## Commands

- `uv run python scripts/internal/rdna4/run_rdna4_profiler_timing_coverage.py ...`
- `uv run python scripts/internal/rdna4/run_rdna4_profiler_sharded_closure.py ...`

## Outputs

- `out/rdna4-validation-reeval-20260613-latest-plus-l2041/profiler-timing-coverage/`
- `out/rdna4-validation-reeval-20260613-latest-plus-l2041/sharded-closure/`
- `out/rdna4-validation-reeval-20260613-latest-plus-l2041/merged/evaluation-summary.md`
- `out/rdna4-validation-reeval-20260613-latest-plus-l2041/merged/evaluation-summary.json`

## Result

- Profiler-backed timing coverage: `131 / 235` problems (`55.7447%`).
- Full profiler-backed timing coverage remains `false`.
- Remaining sharded closure targets: `63`.
- Status counts from the regenerated scripts/internal report:
  - `profiler_backed`: `131`
  - `partial_profiler_backed`: `56`
  - `ready_missing_profiler_timing`: `22`
  - `hardware_evidence_deferred`: `19`
  - `profiler_blocked`: `7`
  - `fallback_timing`: `0`

## Verification

- Parsed `merged/evaluation-summary.json` with `uv run python -m json.tool`.
- Parsed `merged/profiler-timing-coverage-summary.json` with
  `uv run python -m json.tool`.
- Parsed `merged/sharded-closure-audit.json` with `uv run python -m json.tool`.
