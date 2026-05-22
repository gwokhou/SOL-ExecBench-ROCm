# Phase 22: RDNA 4 Validation Closure - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning
**Mode:** Auto-generated (autonomous validation phase)

<domain>

## Phase Boundary

Phase 22 validates v1.4 on the visible RDNA 4 environment and records durable
unit + E2E evidence. It closes RDNA 4 validation for this milestone while still
avoiding CDNA 3 hardware-validation claims.

</domain>

<decisions>

## Implementation Decisions

### the agent's Discretion

- Use the existing `sol-execbench` CLI benchmark flow for E2E evidence.
- Use focused unit tests that cover v1.4 compatibility, diagnostics, derived
  evidence, CDNA 3 readiness guardrails, and support docs.
- Record raw trace JSONL and a concise validation evidence document in the phase
  directory.
- Do not introduce new public interfaces.

</decisions>

<code_context>

## Existing Code Insights

- RDNA 4 is visible locally as `gfx1200` through ROCm/PyTorch.
- `tests/sol_execbench/test_e2e.py` covers package/eval-driver E2E behavior.
- `examples/pytorch/linear_backward` provides a small existing CLI benchmark
  sample suitable for `sol-execbench` validation.

</code_context>

<specifics>

## Specific Ideas

- Run focused v1.4 unit tests.
- Run `uv run pytest tests/sol_execbench/test_e2e.py`.
- Run `uv run sol-execbench examples/pytorch/linear_backward --solution ... --output ... --json`.
- Validate the emitted trace JSONL has 3 `PASSED` records.

</specifics>

<deferred>

## Deferred Ideas

CDNA 3 hardware validation remains future scope.

</deferred>
