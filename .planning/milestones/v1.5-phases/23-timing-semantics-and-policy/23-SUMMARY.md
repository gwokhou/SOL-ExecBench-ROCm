# Phase 23: Timing Semantics and Policy - Summary

**Status:** Executed  
**Completed:** 2026-05-22  
**Plan:** `23-PLAN.md`  
**Requirements:** TIME-01, TIME-02, TIME-03, TIME-04

## Delivered

- Added `src/sol_execbench/core/bench/timing_policy.py` with pure source
  classification and timing policy models.
- Mapped public `SupportedLanguages` values to internal timing sources:
  `pytorch`, `triton`, `hip_native`, `mixed`, and `unknown`.
- Defined policy metadata for backend, activity domain, aggregation rule,
  interpretation, fallback status, and reason.
- Added `docs/user/rocm_timing.md` documenting the
  `source_type -> timer_backend -> interpretation` chimney model.
- Added focused unit and documentation audit tests for Phase 23 timing
  semantics.

## Public Interface Impact

None. Canonical trace JSONL, eval-driver timing execution, solution schema, and
CLI behavior are unchanged. Phase 23 defines timing semantics only; profiler
collection and default timing replacement remain Phase 24 work.

## Verification

```bash
uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py
uv run pytest tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/bench/timing_policy.py tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py
```

Result: passed.

## Commits

- `ce36a52` - `feat: add ROCm timing policy semantics`
