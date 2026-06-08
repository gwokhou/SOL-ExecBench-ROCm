---
phase: 164
status: complete
completed_at: "2026-06-08"
---

# Phase 164 Summary: RDNA4 Memory Readiness Classifier Hardening

## Result

Hardened RDNA4 memory/readiness blocker classification while preserving the
coarse denominator status boundary. Memory OOM blockers still count as
`reference_oom_blocked` for denominator totals, but reports now retain detailed
blocker classes for scheduling and audit.

## Implementation

- Added `blocker_class_counts` to profiler timing coverage reports and summary
  output.
- Distinguished `reference_oom_blocked`, `gen_inputs_oom_blocked`,
  `user_solution_oom`, `memory_oom_with_profiler_gap`, and timeout blockers.
- Extended the partial failure ledger with timeout detection and blocker
  decisions.
- Added CPU-safe regression tests for detailed blocker classes.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_partial_failures.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check src/sol_execbench/core/dataset/profiler_timing_coverage.py scripts/run_rdna4_profiler_timing_coverage.py scripts/run_rdna4_profiler_partial_failures.py tests/sol_execbench/test_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_partial_failures.py`
