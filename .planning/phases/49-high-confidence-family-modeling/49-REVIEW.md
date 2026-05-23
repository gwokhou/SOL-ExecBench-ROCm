---
phase: 49-high-confidence-family-modeling
reviewed: 2026-05-23T07:19:56Z
depth: deep
files_reviewed: 8
files_reviewed_list:
  - src/sol_execbench/core/scoring/amd_bound_graph.py
  - src/sol_execbench/core/scoring/amd_bound_estimates.py
  - src/sol_execbench/core/scoring/solar_derivation.py
  - tests/sol_execbench/test_amd_bound_graph.py
  - tests/sol_execbench/test_amd_bound_estimates.py
  - tests/sol_execbench/test_solar_derivation_evidence.py
  - tests/sol_execbench/test_solar_derivation_family_modeling.py
  - tests/sol_execbench/test_public_contract_guardrails.py
findings:
  critical: 2
  warning: 0
  info: 0
  total: 2
status: issues_found
---

# Phase 49: Code Review Report

**Reviewed:** 2026-05-23T07:19:56Z
**Depth:** deep
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Reviewed Phase 49 commits from `544dba9` through `6eb3317`, the phase plans/summaries, and the requested scoring modules/tests. The public schema, CLI, parser exact-key checks, and AMD SOL v1/v2 boundary guardrails are covered, but two modeling defects remain in the new high-confidence evidence path. Both affect formula/byte correctness and should be fixed before this phase is treated as complete.

## Critical Issues

### CR-01: 3D Linear Projections Lose Formula Evidence

**File:** `src/sol_execbench/core/scoring/amd_bound_estimates.py:989`

**Issue:** `_infer_gemm_dims()` only supports a 2D RHS in the pure 2D case. A normal transformer-style `F.linear(x, weight)` with `x.shape == (B, S, K)`, `weight.shape == (N, K)`, and `out.shape == (B, S, N)` is classified as `linear_projection`, but `_infer_gemm_dims()` returns `None`, so `_gemm_estimate()` emits empty formula inputs, `flops=0.0`, `confidence=INEXACT`, and missing-shape warnings. This violates Phase 49's linear projection contract for a common public workload shape and undercuts SOLAR sidecar formula/bound evidence.

**Fix:**

```python
if len(lhs_shape) >= 3 and len(rhs_shape) == 2 and len(out_shape) >= 3:
    return {
        "B": int(prod(out_shape[:-2])),
        "M": int(out_shape[-2]),
        "N": int(out_shape[-1]),
        "K": int(lhs_shape[-1]),
    }
```

Add a regression test that builds a `Definition` using `F.linear` over `["B", "S", "K"]` input and `["N", "K"]` weight, then asserts `formula_kind == "batched_gemm_flops"`, `formula_inputs == {"B": B, "M": S, "N": N, "K": K}`, nonzero FLOPs, and supported confidence.

### CR-02: Rotary-Like Byte Estimates Double Count Memory Traffic

**File:** `src/sol_execbench/core/scoring/amd_bound_estimates.py:568`

**Issue:** `_visible_memory_estimate()` computes `read_bytes + write_bytes`, then for `rotary_like` sets `movement_bytes` to the same value and adds it again to `total_bytes`. A rotary-like elementwise operation already accounts for visible reads and writes through those fields; adding an extra movement bucket doubles the byte evidence and inflates `memory_bound_ms` in the group-local bound evidence. The current tests only assert positive bytes, so this regression is not caught.

**Fix:**

```python
movement_bytes = 0.0
return OperatorWorkEstimate(
    ...
    movement_bytes=movement_bytes,
    total_bytes=read_bytes + write_bytes,
    ...
)
```

If rotary-like movement is meant to be modeled separately, document that contract and avoid including the same bytes in both read/write and movement totals. Add an exact-byte regression for `(x * cos) + (x * sin)` that asserts `total_bytes == read_bytes + write_bytes` for each rotary-like estimate.

---

_Reviewed: 2026-05-23T07:19:56Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
