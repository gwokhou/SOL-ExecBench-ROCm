# Phase 24: rocprofv3 Default Timing Path - Research

**Researched:** 2026-05-22
**Domain:** ROCm profiler-backed timing collection and evidence parsing
**Confidence:** MEDIUM-HIGH

<research_summary>
## Summary

Phase 24 should introduce profiler-backed timing as a controlled external
collector/parser and a policy-aware default backend selector. `rocprofv3` is
available locally and supports `--kernel-trace`, `--hip-runtime-trace`, output
format selection, output directory, output file naming, and application
execution after `--`.

The safest implementation path is not to splice an external profiler directly
inside `time_runnable()`'s per-call event loop. Instead, add a small
`rocm_profiler.py` module that can build `rocprofv3` commands, parse
representative CSV outputs, aggregate kernel activity durations, and emit a
derived timing evidence payload. Then add policy-aware selection helpers that
mark profiler-backed timing as default when `TimingPolicy.backend` is
`rocprofv3` and a profiler is available; otherwise return an explicit fallback
policy.

**Primary recommendation:** Implement `rocprofv3` wrapper/parser/evidence
models and policy-aware backend selection with fixture tests. Keep canonical
trace JSONL and eval-driver correctness semantics unchanged.
</research_summary>

<architecture_patterns>
## Architecture Patterns

```
TimingPolicy from Phase 23
    |
    v
resolve profiler default
    |-- rocprofv3 available + policy backend rocprofv3 -> profiler-backed path
    |-- unavailable / unsupported / PyTorch attribution-only -> explicit fallback
    |
    v
Rocprofv3 command builder
    -> controlled output directory
    -> kernel trace + HIP runtime trace
    -> csv output
    -> application command after --
    |
    v
CSV parser
    -> normalized rows
    -> domain filtering
    -> kernel activity rows
    -> aggregate duration
    |
    v
Timing evidence dict
```

## Recommended Files

- `src/sol_execbench/core/bench/rocm_profiler.py`
- `tests/sol_execbench/test_rocm_profiler.py`
- `docs/user/rocm_timing.md`
- `tests/sol_execbench/test_rocm_eval_timing_audit.py`

## Parser Strategy

`rocprofv3` CSV fields can vary by ROCm version and output mode. The parser
should normalize column names by lowercasing and removing punctuation, then
look for duration, start/end timestamp, kernel name, and domain/type columns.
Fixture tests should cover representative kernel and HIP runtime rows rather
than depending on live hardware traces.

## Pitfalls

- Do not parse HIP runtime/API duration as kernel activity.
- Do not silently fall back to event timing.
- Do not store profiler evidence in canonical trace JSONL.
- Do not include Triton JIT/autotune/setup overhead in steady-state device time
  unless the evidence explicitly says so.
</architecture_patterns>

<validation_architecture>
## Validation Architecture

| Requirement | Validation Type | Test |
|-------------|-----------------|------|
| PROF-01 | Unit | Parser fixture produces kernel rows and aggregate duration. |
| PROF-02 | Unit | Default backend selector chooses profiler-backed timing when policy backend is `rocprofv3` and available. |
| PROF-03 | Unit | Fallback evidence includes backend, reason, and interpretation. |
| PROF-04 | Unit/docs | Evidence dict includes tool version, GPU arch, activity domain, aggregation rule, and parsed rows. |

**Quick command:** `uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_rocm_eval_timing_audit.py`

**Full command:** `uv run pytest tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_rocm_eval_timing_audit.py tests/sol_execbench/test_public_contract_guardrails.py`
</validation_architecture>

<sources>
## Sources

- `rocprofv3 --help` on local ROCm environment.
- `src/sol_execbench/core/bench/timing_policy.py` - Phase 23 timing policy.
- `src/sol_execbench/core/reporting.py` - derived evidence pattern.
- ROCprofiler-SDK `rocprofv3` docs:
  https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
</sources>

---
*Phase: 24-rocprofv3-default-timing-path*
*Research completed: 2026-05-22*
*Ready for planning: yes*
