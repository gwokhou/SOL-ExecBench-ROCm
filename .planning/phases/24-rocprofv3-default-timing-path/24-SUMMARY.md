# Phase 24: rocprofv3 Default Timing Path - Summary

**Status:** Executed  
**Completed:** 2026-05-22  
**Plan:** `24-PLAN.md`  
**Requirements:** PROF-01, PROF-02, PROF-03, PROF-04

## Delivered

- Added `src/sol_execbench/core/bench/rocm_profiler.py`.
- Added `rocprofv3` command construction for kernel and HIP runtime traces with
  controlled CSV output paths.
- Added robust fixture-based CSV parsing for kernel and HIP runtime/API rows.
- Added derived profiler timing evidence payloads with tool version, GPU
  architecture, activity domain, aggregation rule, parsed rows, backend,
  interpretation, and fallback metadata.
- Added policy-aware default timing selection that chooses profiler-backed
  timing for `rocprofv3` policies when available and explicit fallback otherwise.
- Extended ROCm timing docs with profiler evidence requirements.

## Public Interface Impact

Canonical trace JSONL remains unchanged. Phase 24 adds internal profiler
evidence helpers and documentation, plus tests. It does not run live profiler
collection inside the eval driver.

## Verification

```bash
uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_rocm_eval_timing_audit.py
uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/bench/rocm_profiler.py tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_rocm_eval_timing_audit.py
```

Result: passed.

## Commits

- `0c3f368` - `feat: add rocprofv3 timing evidence helpers`
