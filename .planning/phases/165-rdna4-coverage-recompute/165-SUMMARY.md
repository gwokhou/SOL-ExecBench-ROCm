---
phase: 165
status: complete
completed_at: "2026-06-08"
---

# Phase 165 Summary: RDNA4 Coverage Recompute

## Result

Recomputed RDNA4 profiler timing coverage under the hardened denominator and
classifier policy. The new report includes a deterministic blocker ledger for
all non-passing/non-profiler-backed rows.

## Implementation

- `scripts/run_rdna4_profiler_timing_coverage.py`
  - Added `blocker-ledger.json` output.
  - Added deterministic ledger checksums.
  - Carries `blocker_class_counts` into `coverage-summary.json`.
- Added CPU-safe tests for blocker ledger generation.

## Real RDNA4 Evidence

Generated `out/rdna4-coverage-recompute-20260608/`:

- `coverage.json`
- `coverage.md`
- `coverage-summary.json`
- `blocker-ledger.json`

Current recomputed denominator:

- `problem_denominator`: 235
- `profiler_backed`: 61
- `reference_oom_blocked`: 10
- `profiler_blocked`: 4
- `timing_fallback`: 46
- `readiness_blocked`: 114
- `blocked_or_non_passing_count`: 174
- `ledger_checksum`: `a4d7231e059992fe0d067496481ec51226c1c7d4d3be6abe7dc8472516d36c13`

Detailed memory blocker classes:

- `reference_oom_blocked`: 6
- `gen_inputs_oom_blocked`: 2
- `user_solution_oom`: 1
- `memory_oom_with_profiler_gap`: 1

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_timing_coverage.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check scripts/run_rdna4_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_timing_coverage.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_rdna4_profiler_timing_coverage.py --output-dir out/rdna4-coverage-recompute-20260608 --timing-evidence-dir out/rdna4-profiler-sharded-closure-l1026-20260608/timing --timing-evidence-dir out/rdna4-profiler-workload-aggregate-20260608-v2/timing --timing-evidence-dir out/rdna4-profiler-backed-timing-full-20260608/timing --timing-evidence-dir out/rdna4-timing-evidence/timing`
