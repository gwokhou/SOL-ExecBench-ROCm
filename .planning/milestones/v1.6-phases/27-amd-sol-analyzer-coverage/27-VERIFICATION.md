---
status: passed
phase: 27
---

# Phase 27 Verification: AMD SOL Analyzer Coverage

**Verified:** 2026-05-22
**Result:** Passed

## Must-Haves

| Requirement | Result | Evidence |
|-------------|--------|----------|
| SOLCOV-01 | Passed | Analyzer now recognizes reductions, normalization-like calls, softmax-like calls, activations, and data movement beyond v1.5 matmul/elementwise. |
| SOLCOV-02 | Passed | Tests verify supported/inexact/unsupported confidence labels and rationales remain visible. |
| SOLCOV-03 | Passed | `AmdSolCoverageSummary` and artifact `coverage_summary` report coverage counts before scoring. |
| SOLCOV-04 | Passed | Bound artifacts retain per-op evidence and tests confirm canonical trace payloads are not mutated. |

## Commands

```bash
uv run pytest tests/sol_execbench/test_amd_sol_bounds.py
uv run ruff check src/sol_execbench/core/scoring/amd_sol.py tests/sol_execbench/test_amd_sol_bounds.py
```

Both commands passed.

## Residual Risk

The new formulas are conservative and intentionally labeled `INEXACT` for most
newly recognized families. Full upstream SOLAR parity remains future scope.
