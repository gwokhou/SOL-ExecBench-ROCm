---
phase: 166
status: complete
completed_at: "2026-06-09"
---

# Phase 166 Summary: RDNA4 Rocprof Timing Closure

## Result

Accepted current-device profiler-closure OOM as an explicit RDNA4 blocker class
for the current 16GB `gfx1200` host. Phase 166 did not increase the
`profiler_backed` numerator, but it reduced the opaque fallback bucket by
promoting observed profiler-closure OOM cases into accounted blocker rows.

## Implementation

- Added `profiler_closure_oom_blocked` classification for HIP OOMs that occur
  during `rocprofv3` replacement correctness/error-stat computation.
- Extended coverage and partial-failure classifiers to parse stderr OOM
  tracebacks.
- Updated `docs/internal/RDNA4-DENOMINATOR-POLICY.md`.
- Regenerated accepted coverage at
  `out/rdna4-coverage-recompute-accepted-20260609/`.

## Evidence

- `out/rdna4-rocprof-timing-closure-20260608-smoke/`
- `out/rdna4-rocprof-timing-closure-20260608-l1028-sharded/`
- `out/rdna4-rocprof-timing-closure-20260608-l1053-sharded/`
- `out/rdna4-rocprof-timing-closure-20260608-l1053-offset9-retry/`
- `out/rdna4-coverage-recompute-accepted-20260609/`

Accepted coverage:

- `problem_denominator`: 235
- `profiler_backed`: 61
- `reference_oom_blocked`: 13
- `profiler_blocked`: 2
- `timing_fallback`: 45
- `readiness_blocked`: 114
- `profiler_closure_oom_blocked`: 4
- `ledger_checksum`: `b45453434a2caff827d831187a27da9bd4e95b218db496361c4cb69bf29b891a`

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_partial_failures.py tests/sol_execbench/test_rdna4_profiler_timing_coverage.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check src/sol_execbench/core/dataset/profiler_timing_coverage.py scripts/run_rdna4_profiler_partial_failures.py tests/sol_execbench/test_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_partial_failures.py tests/sol_execbench/test_rdna4_profiler_timing_coverage.py docs/internal/RDNA4-DENOMINATOR-POLICY.md`
- `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_rdna4_profiler_timing_coverage.py --output-dir out/rdna4-coverage-recompute-accepted-20260609 --timing-evidence-dir out/rdna4-rocprof-timing-closure-20260608-l1053-sharded/timing --timing-evidence-dir out/rdna4-rocprof-timing-closure-20260608-l1028-sharded/timing --timing-evidence-dir out/rdna4-profiler-sharded-closure-l1026-20260608/timing --timing-evidence-dir out/rdna4-profiler-workload-aggregate-20260608-v2/timing --timing-evidence-dir out/rdna4-profiler-backed-timing-full-20260608/timing --timing-evidence-dir out/rdna4-timing-evidence/timing`
