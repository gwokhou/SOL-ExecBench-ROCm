---
phase: 50-degraded-complex-family-modeling
reviewed: 2026-05-23T08:24:14Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - src/sol_execbench/core/scoring/amd_bound_graph.py
  - src/sol_execbench/core/scoring/amd_bound_estimates.py
  - src/sol_execbench/core/scoring/solar_derivation.py
  - tests/sol_execbench/test_amd_bound_graph.py
  - tests/sol_execbench/test_amd_bound_estimates.py
  - tests/sol_execbench/test_solar_derivation_evidence.py
  - tests/sol_execbench/test_solar_derivation_family_modeling.py
  - tests/sol_execbench/test_public_contract_guardrails.py
  - tests/sol_execbench/test_amd_sol_v2.py
findings:
  critical: 2
  warning: 0
  info: 0
  total: 2
status: issues_found
---

# Phase 50: Code Review Report

**Reviewed:** 2026-05-23T08:24:14Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Reviewed the Phase 50 diff after `12b1e4a`, with emphasis on MoE route evidence, SSM/Mamba recurrence evidence, deterministic warning/formula names, degraded/unsupported status handling, and public boundary guardrails. The public schema/CLI/trace guardrails are covered, and the new deterministic formula/warning names are tested, but two source-backed evidence rules still have correctness holes that can fabricate scoring inputs.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: MoE dispatch uses the first graph-wide top-k instead of the dispatch's route tensor

**Classification:** BLOCKER

**File:** `src/sol_execbench/core/scoring/amd_bound_graph.py:873`

**Issue:** `_annotate_moe_graph` finds the first MoE `top_k` node in the entire graph and copies that `route_top_k` onto every dispatch node that lacks local `route_top_k` metadata. This does not verify that the dispatch consumes the selected top-k output. A reference with an unused `torch.topk(scores, k=1)` followed by the actual dispatch route `torch.topk(scores, k=4)` marks the dispatch as `route_top_k=1`, then `amd_bound_estimates.py:516` computes `moe_static_route_flops` with the wrong route cardinality and reports supported static evidence. This violates the Phase 50 no-fabrication requirement for top_k and token-to-expert route assignment.

**Fix:** Bind `route_top_k` only through graph dataflow. For each dispatch node, identify the input tensor representing route/gate assignment, resolve its producer node, and copy `route_top_k` only when that producer is the matching `top_k` node. If the route tensor producer is absent or ambiguous, keep the dispatch degraded with `route:top_k` and `route:static_cardinality` missing.

```python
route_producer = _producer_node_for_input(graph, nodes, node, input_index=2)
if (
    route_producer is not None
    and route_producer.op_family == OpFamily.MOE
    and route_producer.attributes.get("subrole") == "top_k"
    and isinstance(route_producer.attributes.get("route_top_k"), int)
):
    attrs["route_top_k"] = route_producer.attributes["route_top_k"]
    attrs["route_cardinality_source"] = f"{route_producer.node_id}.topk.k"
else:
    attrs["missing_route_metadata"] = ("route:top_k", "route:static_cardinality")
```

Add a regression test with two visible `topk` calls where only the second feeds `dispatch_and_combine`, and assert the dispatch uses the second value. Add a second test where a visible top-k is unrelated to dispatch and assert the dispatch degrades instead of inheriting it.

### CR-02: SSM/Mamba static scan fabricates batch=1 when batch shape evidence is missing

**Classification:** BLOCKER

**File:** `src/sol_execbench/core/scoring/amd_bound_estimates.py:699`

**Issue:** `_ssm_visible_formula_inputs` inserts `batch=1` whenever `sequence_length` is present and no input batch shape or explicit `batch_size` exists. That default makes `_ssm_missing_static_scan_inputs` believe all static scan inputs are present, so `_ssm_mamba_estimate` emits `ssm_mamba_static_scan_flops` and positive FLOPs even when the scan input shape is missing. The estimate may still be `inexact` because byte metadata is missing, but the formula kind and formula inputs incorrectly claim static recurrence math with a fabricated batch dimension.

**Fix:** Remove the implicit batch fallback. Treat missing batch as missing static scan evidence and use `ssm_mamba_degraded_scan_bytes` until the batch dimension is visible from an input tensor or an explicitly source-backed attribute.

```python
if input_tensors and input_tensors[0].shape:
    formula_inputs["batch"] = int(input_tensors[0].shape[0])
elif isinstance(node.attributes.get("batch_size"), int):
    formula_inputs["batch"] = int(node.attributes["batch_size"])
```

Add a regression test where a recognized scan has `sequence_length`, `hidden_size`, and `state_shape`, but the scan input tensor shape is `None`. The expected result should be `ssm_mamba_degraded_scan_bytes`, no `batch` formula input, `shape:batch` missing evidence, and no static scan FLOPs.

---

_Reviewed: 2026-05-23T08:24:14Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
