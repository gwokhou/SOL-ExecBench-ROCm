---
status: completed
quick_id: 260613-promote-l2041-merged
slug: promote-l2041-merged
description: Promote the latest L2/041-closed RDNA4 coverage into canonical merged artifacts
created_at: 2026-06-13T20:26:44+08:00
---

# Quick Task 260613-promote-l2041-merged

## Goal

Complete P0: promote the latest L2/041-closed coverage into a canonical
`merged/` artifact directory so downstream assessment no longer reads the
superseded `130/235` summary with one partial problem.

## Inputs

- Latest coverage:
  `out/rdna4-validation-reeval-20260613-latest-plus-l2041/profiler-timing-coverage/`
- Previous merged static reports:
  `out/rdna4-validation-reeval-20260613-latest/merged/`

## Expected Output

- Canonical merged directory:
  `out/rdna4-validation-reeval-20260613-latest-plus-l2041/merged/`
- Headline:
  - `profiler_backed`: `131 / 235`
  - `partial_profiler_backed`: `0`
  - `ready_missing_profiler_timing`: `0`

## Plan

1. [x] Rebuild `merged/` using the latest coverage artifacts.
2. [x] Regenerate `evaluation-summary.json`, `evaluation-summary.md`, and
   `sharded-closure-audit.json` from the latest coverage.
3. [x] Verify canonical counts and ensure L2/041 is no longer listed as a
   remaining sharded closure target.

## Result

Completed.

New canonical merged directory:

- `out/rdna4-validation-reeval-20260613-latest-plus-l2041/merged/`

Canonical headline:

- `profiler_backed`: `131 / 235`
- `profiler_backed_coverage_pct`: `55.7447`
- `partial_profiler_backed`: `0`
- `ready_missing_profiler_timing`: `0`

Remaining sharded closure targets:

- 6 `profiler_blocked` problems
- `L2/041_kv_shared_attention_with_dual_rope` is no longer listed.

Validation:

- `uv run python .planning/quick/260613-promote-l2041-merged/merge_latest_plus_l2041.py`
- Python consistency assertion:
  `merged validation ok`
