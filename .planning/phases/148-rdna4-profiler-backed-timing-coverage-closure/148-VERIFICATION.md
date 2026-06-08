# Phase 148 Verification

## Commands

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_profiler_timing_coverage.py -q
```

Result: `3 passed`.

```text
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check src/sol_execbench/core/dataset/profiler_timing_coverage.py scripts/run_rdna4_profiler_timing_coverage.py tests/sol_execbench/test_profiler_timing_coverage.py
```

Result: `All checks passed`.

```text
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/run_rdna4_profiler_timing_coverage.py
```

Result: generated `out/rdna4-profiler-timing-coverage/coverage.json`,
`coverage-summary.json`, and `coverage.md`.

## Current Coverage Baseline

- Problem denominator: 235
- Profiler-backed problems: 0
- Fallback timing problems: 121
- Ready missing profiler timing problems: 0
- Readiness-blocked problems: 114
- Full profiler-backed timing coverage: false

## Interpretation

The full 235-problem denominator is now accounted for, but the current archived
RDNA4 timing sidecars do not satisfy profiler-backed coverage. The next phase
should execute expanded direct-`eval_driver.py` `rocprofv3` timing batches and
replace fallback sidecars with profiler-backed timing evidence where feasible.
