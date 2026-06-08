# Phase 149 Verification

## Commands

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_timing_batch.py -q
```

Result: `29 passed`.

```text
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check src/sol_execbench/core/bench/rocm_profiler.py src/sol_execbench/core/dataset/profiler_timing_coverage.py scripts/run_rdna4_profiler_timing_batch.py tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_profiler_timing_coverage.py tests/sol_execbench/test_rdna4_profiler_timing_batch.py
```

Result: `All checks passed`.

```text
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/run_rdna4_profiler_timing_batch.py --limit 1 --workload-limit 1 --timeout 300 --output-dir out/rdna4-profiler-backed-timing-batch-real-smoke
```

Result: passed outside the sandbox and produced a real
`_kernel_trace.csv`. This proves the forced direct-`eval_driver.py`
`rocprofv3` path works for a bounded single-workload smoke, but the sidecar is
not a full problem replacement because `--workload-limit 1` was applied.

```text
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/run_rdna4_profiler_timing_batch.py --limit 1 --timeout 600 --output-dir out/rdna4-profiler-backed-timing-batch-real-full-smoke-v2
```

Result: exited `1` outside the sandbox after producing valid kernel trace
artifacts. The replacement sidecar was marked incomplete because the first
target problem produced trace status counts:

- `PASSED`: 19
- `INVALID_REFERENCE`: 1

```text
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/run_rdna4_profiler_timing_coverage.py --output-dir out/rdna4-profiler-backed-timing-batch-real-full-smoke-v2/coverage --timing-evidence-dir out/rdna4-profiler-backed-timing-batch-real-full-smoke-v2/timing --timing-evidence-dir out/rdna4-timing-evidence/timing
```

Result: coverage remained at `0` profiler-backed problems, `121` fallback
timing problems, and `114` readiness-blocked problems. The incomplete
replacement sidecar did not pollute the full-problem coverage count.

## Interpretation

Phase 149 has a working replacement runner and guardrails against false
profiler-backed success. Full 121-problem replacement is not complete: the
first full-problem target reproduced a profiler-run OOM/`INVALID_REFERENCE`
under full workload coverage.
