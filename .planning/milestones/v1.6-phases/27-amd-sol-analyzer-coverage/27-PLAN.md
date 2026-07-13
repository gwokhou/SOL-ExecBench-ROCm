# Phase 27 Plan: AMD SOL Analyzer Coverage

**Created:** 2026-05-22
**Status:** Ready for execution
**Requirements:** SOLCOV-01, SOLCOV-02, SOLCOV-03, SOLCOV-04

## Goal

Broaden AMD SOL/SOLAR-like analyzer coverage and make coverage confidence
visible before scoring, while keeping all SOL evidence as derived artifacts
outside canonical trace JSONL.

## Tasks

### 27-01: Add coverage summary artifact

**Files:**
- `src/sol_execbench/core/scoring/amd_sol.py`
- `tests/sol_execbench/test_amd_sol_bounds.py`

**Work:**
- Add a derived coverage summary dataclass or helper that counts supported,
  inexact, and unsupported graph/work estimates.
- Include operation-type counts and a `to_dict()` payload suitable for score
  workflows.
- Expose it without changing public trace models.

**Verification:**
- Unit test summary counts and derived serialization.

### 27-02: Refactor analyzer classification

**Files:**
- `src/sol_execbench/core/scoring/amd_sol.py`
- `tests/sol_execbench/test_amd_sol_bounds.py`

**Work:**
- Replace or isolate `_GraphVisitor` string checks behind explicit analyzer
  metadata/helpers.
- Preserve existing matmul and elementwise behavior.
- Add clear op type names for reductions, normalization, softmax-like,
  activation, and data-movement nodes.

**Verification:**
- Existing matmul/elementwise/unsupported tests still pass.
- New tests assert expected op types and confidence labels.

### 27-03: Extend conservative work estimates

**Files:**
- `src/sol_execbench/core/scoring/amd_sol.py`
- `tests/sol_execbench/test_amd_sol_bounds.py`

**Work:**
- Estimate reductions and normalization-like patterns conservatively.
- Estimate softmax-like and activation families as `INEXACT` unless exact
  semantics are known.
- Estimate data-movement nodes with zero or low FLOP and byte rationale.
- Keep unsupported operations explicit.

**Verification:**
- Tests cover reductions, data movement, softmax-like or activation patterns,
  and unsupported preservation.

### 27-04: Add compatibility and documentation coverage

**Files:**
- `docs/internal/analysis.md`
- `tests/sol_execbench/test_amd_sol_bounds.py`

**Work:**
- Document AMD SOL coverage semantics and confidence labels.
- Add a focused test showing SOL artifacts and summaries do not mutate trace
  model payloads.

**Verification:**
- Run `uv run pytest tests/sol_execbench/test_amd_sol_bounds.py`.
- Run `uv run ruff check src/sol_execbench/core/scoring/amd_sol.py tests/sol_execbench/test_amd_sol_bounds.py`.

## Plan Check

This plan maps all Phase 27 requirements:

| Requirement | Task |
|-------------|------|
| SOLCOV-01 | 27-02, 27-03 |
| SOLCOV-02 | 27-02, 27-03 |
| SOLCOV-03 | 27-01 |
| SOLCOV-04 | 27-01, 27-03, 27-04 |

## Risks

- Overstating analytical precision. Mitigation: default to `INEXACT` with
  rationale unless support is exact.
- Breaking existing score artifacts. Mitigation: preserve existing function
  names and dataclass fields, only add derived helpers.
- Accidentally touching public trace/schema behavior. Mitigation: add focused
  trace immutability test.
