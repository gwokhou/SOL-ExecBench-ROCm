# Phase 19: Compatibility and Practice Inventory - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Auto-generated (autonomous infrastructure phase)

<domain>

## Phase Boundary

Phase 19 delivers a source-grounded compatibility inventory and guardrail plan
for v1.4. It classifies hip-execbench practices as accepted, rejected, or
deferred before implementation and does not introduce runtime dependencies,
public API changes, or benchmark execution changes.

</domain>

<decisions>

## Implementation Decisions

### the agent's Discretion

- Treat Phase 19 as an infrastructure/contract phase; use roadmap,
  requirements, and research as the spec.
- Preserve public CLI behavior, Pydantic schemas, solution format, trace JSONL,
  and eval-driver semantics.
- Use local source evidence from both repositories; do not rely on README/docs
  claims as implementation evidence.
- Prefer tests and planning/docs artifacts over production runtime changes in
  this phase.

</decisions>

<code_context>

## Existing Code Insights

### Reusable Assets

- Existing guardrail tests in `tests/sol_execbench/test_public_contract_guardrails.py`.
- Existing practice map/doc context in `docs/internal/hip_execbench_practice_map.md`.
- Current schemas under `src/sol_execbench/core/data/`.
- Current CLI in `src/sol_execbench/cli/main.py`.
- Eval semantics in `src/sol_execbench/driver/templates/eval_driver.py` and
  timing/reward-hack helpers.

### Established Patterns

- Public contract protection is test-driven.
- Pydantic v2 models define schema behavior; avoid public field churn.
- CLI errors use Click and trace/eval failures use `EvaluationStatus`.
- Subprocesses use argument lists and eval stdout stays reserved for JSON
  traces.
- New docs/tests should avoid broad refactors.

### Integration Points

- Requirements COMPAT-01..03.
- Roadmap Phase 19 success criteria.
- `hip-execbench` source paths under
  `~/PyCharmMiscProject/hip-playground/hip-execbench/src/`.

</code_context>

<specifics>

## Specific Ideas

No specific user-facing behavior; phase output should be maintainable evidence
and guardrails that enable later phases.

</specifics>

<deferred>

## Deferred Ideas

Implementation of stage diagnostics, evidence helpers, CDNA readiness, and
RDNA4 E2E validation belongs to phases 20-22.

</deferred>
