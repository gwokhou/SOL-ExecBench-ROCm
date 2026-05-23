---
phase: 49-high-confidence-family-modeling
reviewed: 2026-05-23T07:23:45Z
depth: deep
files_reviewed: 2
files_reviewed_list:
  - src/sol_execbench/core/scoring/amd_bound_estimates.py
  - tests/sol_execbench/test_amd_bound_estimates.py
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
status: clean
---

# Phase 49: Code Review Report

**Reviewed:** 2026-05-23T07:23:45Z
**Depth:** deep
**Files Reviewed:** 2
**Status:** clean

## Summary

Re-reviewed fix commit `b383ccf` for the two blockers previously reported in Phase 49. The changed estimator logic and regression tests now address both defects.

CR-01 is fixed: rank-3-or-higher linear projection input with a rank-2 weight now infers batched GEMM dimensions, emits `batched_gemm_flops`, records `{"B", "M", "N", "K"}` formula inputs, produces nonzero FLOPs, and keeps supported confidence when tensor evidence is complete.

CR-02 is fixed: rotary-like visible memory estimates now report `total_bytes == read_bytes + write_bytes` and leave `movement_bytes` at zero, avoiding the previous duplicate byte bucket.

Regression coverage was added in `tests/sol_execbench/test_amd_bound_estimates.py` for both cases. All reviewed files meet quality standards. No remaining issues found.

## Narrative Findings (AI reviewer)

No Critical, Warning, or Info findings.

## Verification

- `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -k "batched_linear_projection or rotary_like" -n 0 -x` -> 2 passed.
- `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0` -> 110 passed.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_estimates.py tests/sol_execbench/test_amd_bound_estimates.py` -> passed.

---

_Reviewed: 2026-05-23T07:23:45Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
