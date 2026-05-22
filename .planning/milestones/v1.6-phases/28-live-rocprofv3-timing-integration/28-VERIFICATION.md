---
status: passed
phase: 28
---

# Phase 28 Verification: Live rocprofv3 Timing Integration

**Verified:** 2026-05-22
**Result:** Passed

## Must-Haves

| Requirement | Result | Evidence |
|-------------|--------|----------|
| PROF-01 | Passed | `collect_rocprofv3_timing()` invokes benchmark/dataset commands through `rocprofv3` when policy selects profiler-backed timing. |
| PROF-02 | Passed | Collection evidence records policy metadata, tool version, GPU architecture, parsed rows, command, CSV path, return code, stdout/stderr, and fallback reason when applicable. |
| PROF-03 | Passed | Tests verify PyTorch does not masquerade as `rocprofv3` kernel activity and fallback remains explicit. |
| PROF-04 | Passed | Documentation requires compile/autotune/warmup/unrelated rows to be excluded or labeled; collection failures and missing CSV output are explicit fallback states. |

## Commands

```bash
uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py
uv run ruff check src/sol_execbench/core/bench/rocm_profiler.py tests/sol_execbench/test_rocm_profiler.py
```

Both commands passed.

## Residual Risk

Unit tests use an injected runner and do not require real `rocprofv3` hardware
execution. Real profiler output validation remains environment-dependent.
