---
quick_id: 260608-l01
slug: rdna4-reference-oom-classifier
status: complete
completed_at: "2026-06-08"
---

# Summary

Implemented RDNA4 reference OOM classification for partial profiler-backed
timing targets.

## Changes

- Extended `scripts/run_rdna4_profiler_partial_failures.py` to parse non-PASSED
  JSONL trace records from timing sidecar `stdout`.
- Added sharded aggregate handling that follows
  `evidence.source_workloads[*].replacement_path` to inspect per-workload slice
  logs.
- Added `blocker_class`, `blocker_class_counts`, and `failure_details` to the
  classification JSON report.
- Updated closure decisions so detected reference/gen_inputs HIP OOM targets
  are routed to `blocked_on_reference_oom` instead of correctness/runtime
  profiler retry queues.
- Regenerated
  `out/rdna4-profiler-partial-failure-classification-20260608/`.

## Result

The 10 remaining `partial_profiler_backed` RDNA4 targets now classify as:

- `blocked_on_reference_oom`: 10
- `reference_oom_blocked`: 9
- `reference_oom_with_profiler_gap`: 1

This preserves the claim boundary: these targets still do not count as full
profiler-backed timing coverage until every expected workload produces a
passing reference trace with usable rocprofv3 kernel activity.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rdna4_profiler_partial_failures.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check scripts/run_rdna4_profiler_partial_failures.py tests/sol_execbench/test_rdna4_profiler_partial_failures.py`
