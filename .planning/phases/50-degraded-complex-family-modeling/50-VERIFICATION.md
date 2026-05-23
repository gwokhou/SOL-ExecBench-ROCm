---
phase: 50-degraded-complex-family-modeling
verified: 2026-05-23T08:35:57Z
status: passed
score: 8/8 must-haves verified
overrides_applied: 0
---

# Phase 50: Degraded Complex Family Modeling Verification Report

**Phase Goal:** Users can derive conservative, explicitly degraded SOLAR evidence for MoE and SSM/Mamba-like structures when static metadata is incomplete.
**Verified:** 2026-05-23T08:35:57Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | MoE evidence covers routing, top-k selection, expert projection, token dispatch, and combine patterns. | VERIFIED | `amd_bound_graph.py` implements `_annotate_moe_graph`, `_moe_route_metadata_from_topk`, and dispatch route metadata; `solar_derivation.py` emits MoE subroles. `test_moe_visible_static_route_nodes_record_subroles_and_metadata` and `test_moe_static_route_sidecar_matches_positive_fixture_contract` assert router/top_k/dispatch/expert_projection/combine. |
| 2 | MoE dynamic/incomplete routing records degraded evidence with missing route metadata and does not fabricate top_k/expert cardinality/token assignment. | VERIFIED | Dynamic dispatch records `missing_route_metadata=("route:top_k", "route:static_cardinality")`; `_moe_estimate` emits `moe_dynamic_route_bytes` with partial visible inputs only. Tests assert no `route_top_k` on dynamic/unrelated routes and no `top_k` formula input. |
| 3 | MoE taxonomy-only patterns remain unsupported/unscored. | VERIFIED | Taxonomy-only MoE calls are converted to `OpFamily.MOE` with `EstimateConfidence.UNSUPPORTED`, `taxonomy_only=True`, and `unsupported_operator:moe_taxonomy_only`; tests assert no fabricated subroles and unsupported fixture status. |
| 4 | SSM/Mamba evidence covers projection, depthwise convolution, scan/state update, gating, and output projection when visible. | VERIFIED | `_annotate_ssm_mamba_graph` promotes scan chains, predecessors, successors, and visible state updates; `_ssm_mamba_subroles` emits input_projection/depthwise_convolution/scan/state_update/gating/output_projection. Positive graph and sidecar tests assert all roles. |
| 5 | SSM/Mamba missing recurrence/state evidence degrades; scan-only does not fabricate state_update. | VERIFIED | Missing recurrence appends `inexact_operator:ssm_missing_recurrence`; `_ssm_state_update_metadata` requires visible state/update parameters before inserting a `state_update` node. Tests assert missing recurrence/custom scan have no `state_update` and use `ssm_mamba_degraded_scan_bytes` or unsupported evidence. |
| 6 | Group-local formula/byte/bound evidence remains sidecar-only. | VERIFIED | Phase 50 formula kinds are stored in `SolarSemanticGroupEvidence.formula_evidence` values; strict parser test accepts them as values and rejects them as top-level schema keys. Public guardrails assert sidecar evidence names are absent from canonical schemas and trace JSONL. |
| 7 | Public schemas, primary CLI, canonical trace JSONL, AMD-native score eligibility, no-new-dependency, and no-candidate-execution boundaries remain intact. | VERIFIED | `test_primary_cli_does_not_expose_v1_10_solar_derivation_options`, canonical trace guardrails, strict parser checks, and degraded complex-family score eligibility tests passed. `pyproject.toml` was not modified by Phase 50; implementation uses existing graph/estimate/sidecar code and static `Definition`/`Workload` tests. |
| 8 | Phase 49 high-confidence behavior remains green. | VERIFIED | Focused gate includes existing attention, convolution, embedding/positional, and linear projection tests in `test_solar_derivation_family_modeling.py`, `test_amd_bound_graph.py`, and `test_amd_bound_estimates.py`; local run passed `137 passed in 1.25s`. |

**Score:** 8/8 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/sol_execbench/core/scoring/amd_bound_graph.py` | MoE and SSM/Mamba graph recognition, subrole attributes, degraded/unsupported annotations | VERIFIED | `gsd-sdk query verify.artifacts` passed for Plans 50-01 and 50-02; manual review found `_annotate_moe_graph`, `_annotate_ssm_mamba_graph`, route metadata, and state-update guards. |
| `src/sol_execbench/core/scoring/amd_bound_estimates.py` | MoE and SSM/Mamba static/degraded estimate dispatch and deterministic formula kinds | VERIFIED | Dispatches `OpFamily.MOE` and `OpFamily.SSM_MAMBA` to family-specific estimators with `moe_static_route_flops`, `moe_dynamic_route_bytes`, `ssm_mamba_static_scan_flops`, and `ssm_mamba_degraded_scan_bytes`. |
| `src/sol_execbench/core/scoring/solar_derivation.py` | MoE/SSM sidecar subroles, missing evidence, family-specific confidence warnings | VERIFIED | `classify_solar_confidence` calls `_moe_confidence_evidence` and `_ssm_mamba_confidence_evidence`; subrole helpers feed group-local sidecar evidence. |
| `tests/sol_execbench/test_amd_bound_graph.py` | Graph-level positive/degraded/unsupported behavior | VERIFIED | Covers MoE static/dynamic/taxonomy-only, route binding, unrelated top-k, SSM full chain, missing recurrence, custom scan, and non-SSM conv/projection boundaries. |
| `tests/sol_execbench/test_amd_bound_estimates.py` | Estimate-level no-fabrication and formula-kind behavior | VERIFIED | Covers MoE static/dynamic/taxonomy-only, SSM static/degraded/custom scan, and missing scan input batch shape no-fabrication. |
| `tests/sol_execbench/test_solar_derivation_family_modeling.py` | End-to-end sidecar fixture contracts and Phase 49 regressions | VERIFIED | MoE/SSM fixture tests and Phase 49 family tests passed in focused gate. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Public schema, CLI, trace JSONL, and score boundary guardrails | VERIFIED | Phase 50 internal evidence names are asserted absent from public surfaces; degraded complex family score eligibility remains existing AMD-native behavior. |
| `tests/sol_execbench/test_solar_derivation_evidence.py` | Strict parser and sidecar-only formula value checks | VERIFIED | Phase 50 formula kinds parse as sidecar values and are rejected as unknown top-level schema keys. |
| `tests/sol_execbench/test_amd_sol_v2.py` | AMD-native artifact/score regression coverage | VERIFIED | Included in focused phase gate. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| MoE graph nodes | MoE semantic group | `router`, `top_k`, `dispatch`, `expert_projection`, `combine` subroles | WIRED | Graph attributes flow into `_moe_subroles`, `_moe_confidence_evidence`, formula/byte/bound sidecar builders, and fixture tests. |
| MoE estimates | `SolarSemanticGroupEvidence` | `formula_evidence`, `byte_evidence`, `bound_evidence`, `missing_evidence`, `warning_prefixes` | WIRED | `moe_static_route_flops` and `moe_dynamic_route_bytes` are converted into group-local sidecar evidence and tested. |
| SSM/Mamba graph nodes | SSM/Mamba semantic group | `input_projection`, `depthwise_convolution`, `scan`, `state_update`, `gating`, `output_projection` subroles | WIRED | Graph promotion feeds `_ssm_mamba_subroles` and recurrence confidence gates; tests assert full and degraded subrole sets. |
| SSM/Mamba estimates | `SolarSemanticGroupEvidence` | Formula/byte/bound evidence and recurrence missing evidence | WIRED | `ssm_mamba_static_scan_flops` and `ssm_mamba_degraded_scan_bytes` flow into sidecar groups; tests assert status, formula kinds, and no fabricated `state_update`. |
| Phase 50 sidecar names | Public benchmark surfaces | Negative public contract assertions | WIRED | Guardrails assert formula kinds/warnings do not appear in `Definition`, `Workload`, `Trace`, primary CLI help, or canonical trace JSONL. |
| Degraded complex family estimates | AMD-native scoring | Existing score eligibility tests | WIRED | `test_degraded_complex_family_score_eligibility_ignores_solar_sidecars` passed and confirms no SOLAR sidecar ref drives eligibility changes. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `amd_bound_graph.py` | `BoundGraphNode.attributes` for MoE route metadata | Static AST/FX graph, tensor shapes, call args, and workload axes | Yes | FLOWING |
| `amd_bound_estimates.py` | `OperatorWorkEstimate.formula_inputs`, bytes, confidence, warnings | `BoundGraph` nodes and tensors from graph extraction | Yes | FLOWING |
| `solar_derivation.py` | `SolarSemanticGroupEvidence` subroles, formula/byte/bound evidence, missing evidence | `estimate_bound_work(build_bound_graph(...))` | Yes | FLOWING |
| Public guardrail tests | Serialized public payloads and CLI help text | Pydantic models, `Trace.model_dump`, Click CLI runner | Yes | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase 49 plus Phase 50 focused gate | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0` | `137 passed in 1.25s` | PASS |
| Ruff over touched source/tests | `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py` | `All checks passed!` | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| None | `find scripts -path '*/tests/probe-*.sh' -type f` and phase doc probe search | No probes declared or discovered for Phase 50 | SKIP |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DERIVE-02 | `50-01-PLAN.md`, `50-03-PLAN.md` | Conservatively recognize MoE routing, top-k selection, expert projection, token dispatch, and combine patterns, with dynamic routing evidence when static cardinality is incomplete. | SATISFIED | MoE graph, estimate, sidecar, degraded dynamic route, taxonomy-only unsupported, and public boundary tests passed. |
| DERIVE-04 | `50-02-PLAN.md`, `50-03-PLAN.md` | Conservatively recognize SSM/Mamba-like projection, depthwise convolution, scan or state update, gating, and output projection patterns, with degraded evidence when recurrence semantics are incomplete. | SATISFIED | SSM/Mamba graph, estimate, sidecar, missing recurrence degradation, custom scan unsupported, and scan/no-state-update tests passed. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/sol_execbench/core/scoring/amd_bound_graph.py` | 437 | `placeholder` | INFO | FX node operation name, not placeholder implementation. |
| `src/sol_execbench/core/scoring/amd_bound_graph.py` | 1068, 1071, 1086, 1099, 1721, 1752 | `return {}` | INFO | Guard-clause empty metadata returns; not user-visible stubs and downstream code degrades instead of fabricating evidence. |

### Human Verification Required

None.

### Gaps Summary

No gaps found. The implementation provides conservative MoE and SSM/Mamba evidence, degrades incomplete dynamic/recurrence cases explicitly, keeps taxonomy-only/opaque cases unscored, preserves sidecar-only/public boundaries, and keeps the focused Phase 49 regression gate green.

---

_Verified: 2026-05-23T08:35:57Z_
_Verifier: the agent (gsd-verifier)_
