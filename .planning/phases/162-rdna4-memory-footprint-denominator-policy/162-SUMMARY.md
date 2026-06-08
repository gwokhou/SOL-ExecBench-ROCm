---
phase: 162
status: complete
completed_at: "2026-06-08"
---

# Phase 162 Summary: RDNA4 Memory-Footprint Denominator Policy

## Result

Implemented `reference_oom_blocked` as a first-class profiler timing coverage
status. Coverage reports now distinguish:

- Complete profiler-backed timing evidence.
- Partial profiler evidence.
- Profiler lifecycle/evidence gaps.
- Reference/gen_inputs OOM blockers on the current 16GB RDNA4 device.
- Fallback timing and readiness-blocked problems.

## Implementation

- `src/sol_execbench/core/dataset/profiler_timing_coverage.py`
  - Added `reference_oom_blocked_problems`.
  - Detects HIP OOM from timing sidecar JSONL `stdout`.
  - Follows workload-sharded aggregate `source_workloads[*].replacement_path`
    to inspect per-slice sidecar logs.
  - Promotes affected targets to `reference_oom_blocked` without incrementing
    `profiler_backed_problems`.

- `scripts/run_rdna4_profiler_timing_coverage.py`
  - Includes `reference_oom_blocked_problems` in `coverage-summary.json`.

- `scripts/run_rdna4_profiler_partial_failures.py`
  - Keeps the partial/reference-OOM classification report compatible with
    promoted `reference_oom_blocked` statuses.

## Real RDNA4 Evidence

Generated:

- `out/rdna4-profiler-memory-denominator-policy-20260608/coverage-summary.json`
- `out/rdna4-profiler-memory-denominator-policy-20260608/coverage.md`
- `out/rdna4-profiler-partial-failure-classification-20260608/`

Current 235-problem denominator accounting:

- `profiler_backed`: 61
- `reference_oom_blocked`: 10
- `profiler_blocked`: 4
- `timing_fallback`: 46
- `readiness_blocked`: 114
- `partial_profiler_backed`: 0

Profiler-backed timing coverage remains `61/235`, or `25.9574%`.
The denominator is accounted, but full profiler-backed timing coverage remains
false because reference OOM-blocked targets do not count as passing profiler
timing evidence.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_partial_failures.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check src/sol_execbench/core/dataset/profiler_timing_coverage.py scripts/run_rdna4_profiler_timing_coverage.py scripts/run_rdna4_profiler_partial_failures.py tests/sol_execbench/test_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_partial_failures.py`
