# Phase 24: rocprofv3 Default Timing Path - Pattern Map

**Mapped:** 2026-05-22

## File Mapping

| Target File | Role | Closest Existing Analog | Pattern |
|-------------|------|-------------------------|---------|
| `src/sol_execbench/core/bench/rocm_profiler.py` | New profiler wrapper/parser/evidence module | `src/sol_execbench/core/diagnostics.py`, `src/sol_execbench/core/reporting.py` | Frozen dataclasses, `str, Enum`, `to_dict()`, explicit fallback fields. |
| `tests/sol_execbench/test_rocm_profiler.py` | New fixture tests | `tests/sol_execbench/test_timing_policy.py` | Pure unit assertions without live hardware dependency. |
| `docs/rocm_timing.md` | Extend timing docs | Existing Phase 23 doc | Add profiler evidence and fallback wording. |
| `tests/sol_execbench/test_rocm_eval_timing_audit.py` | Documentation audit | Existing audit file | Add string assertions, do not weaken residue checks. |

## Reusable Patterns

- `ProfilerReadiness` in `diagnostics.py` already carries backend, reason,
  fallback, and effective level.
- `DerivedEvidenceReport` in `reporting.py` carries schema version, derived flag,
  canonical output, and a JSON-serializable `to_dict()`.
- `TimingPolicy` from Phase 23 carries source/backend/domain/aggregation and
  interpretation.

## Guardrails

- Parser tests should use fixture CSV strings, not live profiler runs.
- `rocprofv3` command builder must put application command after `--`.
- Evidence must mark itself derived and identify `trace_jsonl` as the canonical
  benchmark output.
- Fallback must be explicit; no hidden event-timing substitution.

---
*Pattern map completed: 2026-05-22*
