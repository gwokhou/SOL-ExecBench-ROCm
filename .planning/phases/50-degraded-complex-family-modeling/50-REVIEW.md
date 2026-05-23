---
phase: 50-degraded-complex-family-modeling
reviewed: 2026-05-23T08:29:42Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - src/sol_execbench/core/scoring/amd_bound_graph.py
  - src/sol_execbench/core/scoring/amd_bound_estimates.py
  - tests/sol_execbench/test_amd_bound_graph.py
  - tests/sol_execbench/test_amd_bound_estimates.py
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
status: clean
---

# Phase 50: Code Review Report

**Reviewed:** 2026-05-23T08:29:42Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** clean

## Summary

Re-reviewed the fixes in `2a161be` for the two previous Phase 50 blockers. CR-01 is fixed: MoE dispatch route cardinality is now resolved from the producer of the consumed dispatch route tensor via `_moe_route_metadata_from_dispatch_input`, and dispatches with unrelated or ambiguous top-k producers remain degraded instead of inheriting graph-wide metadata. CR-02 is fixed: SSM/Mamba static scan formula inputs no longer fabricate `batch=1`; missing scan input batch shape degrades to `ssm_mamba_degraded_scan_bytes` without a `batch` formula input.

Targeted regressions cover multiple top-k producers, unrelated top-k producers, and missing SSM scan input batch shape. No remaining blocker, warning, or info findings were found in the reviewed fix scope.

Verification:

- `uv run pytest tests/sol_execbench/test_amd_bound_graph.py::test_moe_dispatch_uses_top_k_from_consumed_route_tensor tests/sol_execbench/test_amd_bound_graph.py::test_moe_dispatch_does_not_inherit_unrelated_top_k tests/sol_execbench/test_amd_bound_estimates.py::test_ssm_mamba_missing_scan_input_shape_does_not_fabricate_batch` - passed, 3 tests.
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py` - passed.
- `uv run pytest tests/` - attempted locally, but failed in unrelated checks outside this fix scope: missing MIOpen dependency, `build_ext.py` import expectation, hip-execbench practice-map docs, CUDA/NVIDIA residue audit, and CDNA3 requirements text.

## Narrative Findings (AI reviewer)

All reviewed files meet quality standards for the Phase 50 blocker re-review. No issues found.

---

_Reviewed: 2026-05-23T08:29:42Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
