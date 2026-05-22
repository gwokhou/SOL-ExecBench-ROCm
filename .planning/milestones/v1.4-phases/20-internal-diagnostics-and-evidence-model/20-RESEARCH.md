# Phase 20 Research: Internal Diagnostics and Evidence Model

## RESEARCH COMPLETE

**Phase:** 20 - Internal Diagnostics and Evidence Model  
**Date:** 2026-05-22  
**Mode:** Inline autonomous research from local source code

## Existing Local Assets

- `src/sol_execbench/core/diagnostics.py`
  - `DiagnosticStage`, `StageDiagnostic`, and `SolExecBenchError` already cover
    stage-aware parse/package/compile/runtime/verify/timing/environment
    diagnostics.
  - `select_profiler_backend` already models ROCm profiler readiness with
    backend, reason, fallback, and effective level.
  - `rocm_tool_diagnostics` and `local_gfx_target` provide environment
    inspection without changing CLI behavior.
- `src/sol_execbench/core/reporting.py`
  - `summarize_traces` and `format_trace_summary` summarize existing `Trace`
    objects without mutating trace JSONL.
- Tests already cover the diagnostics and trace-summary non-mutation patterns.

## hip-execbench Pattern To Adapt

- `src/errors/index.ts`: typed stage errors with actionable hints.
- `src/profiler/router.ts`: pure profiler backend readiness decision with
  reason and fallback metadata.
- `src/agent/builder.ts`: pure transformation layer from pipeline internals to a
  stable machine-readable report.

## Gap For Phase 20

The repository has internal diagnostics and trace summaries, but it lacks a
single derived evidence/report object that:

- combines trace summary and diagnostics;
- labels itself as derived/non-canonical;
- can be serialized by tests or future tooling without mutating trace JSONL;
- avoids adding public CLI or runtime dependencies.

## Recommended Implementation

- Add dataclasses in `src/sol_execbench/core/reporting.py`:
  - `DerivedEvidenceReport`
  - optional compact diagnostic row serialization
- Add `build_evidence_report(traces, diagnostics=None)` as a pure helper.
- Include constants/fields:
  - `schema_version`
  - `derived=True`
  - `canonical_output="trace_jsonl"`
  - `summary`
  - `diagnostics`
- Add focused tests in existing test files.
- Preserve CLI help guardrails from Phase 19.

## Validation Architecture

```bash
uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/reporting.py tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py
```

## Risks

- Evidence objects could be mistaken for trace output. Mitigation: explicit
  `derived=True` and `canonical_output="trace_jsonl"` fields.
- A future CLI may expose reports prematurely. Mitigation: Phase 20 keeps helper
  APIs internal and tests public CLI help.
