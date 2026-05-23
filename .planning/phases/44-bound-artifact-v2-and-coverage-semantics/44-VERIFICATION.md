---
phase: 44
slug: bound-artifact-v2-and-coverage-semantics
status: passed
verified: 2026-05-23
requirements:
  - BOUND-01
  - BOUND-02
  - BOUND-03
  - BOUND-04
---

# Phase 44 Verification

## Result

Passed.

## Scope Verified

- AMD SOL bound artifact v2 sidecars serialize and load through a strict schema
  contract.
- V2 artifacts contain graph payloads, rich operator work estimates, per-op
  SOL bounds, aggregate bound state, hardware model references, warnings, and
  coverage summaries.
- Per-operation bounds expose compute, memory, SOL, limiting resource,
  confidence, and rationale.
- Coverage summaries report supported, inexact, and unsupported counts by
  operation family plus worst confidence.
- Unsupported evidence forces an `unscored` aggregate state and deterministic
  warnings.
- V1 bound artifacts, primary CLI help, and canonical data schemas remain free
  of v2 sidecar-only fields.

## Automated Verification

- `uv run pytest tests/sol_execbench/test_amd_sol_v2.py -x`
  - Passed: 7 tests.
- `uv run pytest tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x`
  - Passed: 29 tests.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_sol_v2.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_public_contract_guardrails.py`
  - Passed.

## Requirement Mapping

| Requirement | Status | Evidence |
| --- | --- | --- |
| BOUND-01 | Passed | `amd_sol_bound_v2_from_dict()` round-trip and malformed payload tests. |
| BOUND-02 | Passed | Per-op bound tests over matmul rich estimates. |
| BOUND-03 | Passed | Family-aware coverage and worst-confidence tests. |
| BOUND-04 | Passed | Unsupported operation aggregate `unscored` and warning tests. |
