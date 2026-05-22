# Phase 20: Internal Diagnostics and Evidence Model - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Auto-generated (autonomous infrastructure phase)

<domain>

## Phase Boundary

Phase 20 adapts hip-execbench diagnostic and report/evidence practices as
internal helpers. It must preserve trace JSONL as canonical benchmark output and
must not add public CLI behavior, schema fields, or mandatory dependencies.

</domain>

<decisions>

## Implementation Decisions

### the agent's Discretion

- Use Phase 19 compatibility inventory as the non-negotiable boundary.
- Prefer dataclasses and pure helpers under `src/sol_execbench/core/`.
- Treat evidence/report objects as derived artifacts, not canonical benchmark
  output.
- Keep normal `sol-execbench` CLI and eval-driver stdout unchanged.

</decisions>

<code_context>

## Existing Code Insights

- `src/sol_execbench/core/diagnostics.py` already contains stage diagnostics,
  tool checks, gfx classification, and profiler-readiness routing.
- `src/sol_execbench/core/reporting.py` already summarizes existing `Trace`
  objects without mutating their schema.
- `tests/sol_execbench/test_rocm_diagnostics_reporting.py` and
  `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py` provide
  the closest test patterns.
- `docs/internal/v1_4_compatibility_inventory.md` defines Phase 20's public
  contract boundary.

</code_context>

<specifics>

## Specific Ideas

- Add a small derived evidence report dataclass that combines trace summary and
  diagnostics.
- Include explicit metadata such as `derived=True` and
  `canonical_output="trace_jsonl"` so agent/report consumers cannot confuse it
  with benchmark output.
- Add tests that prove trace objects are not mutated and CLI help remains
  unchanged.

</specifics>

<deferred>

## Deferred Ideas

Public report commands, repeated-sample statistical evidence, and hardware
validation evidence belong to later decisions/phases.

</deferred>
