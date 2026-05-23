---
phase: 49-high-confidence-family-modeling
verified: 2026-05-23T07:27:20Z
status: passed
score: 12/12 must-haves verified
overrides_applied: 0
---

# Phase 49: High-Confidence Family Modeling Verification Report

**Phase Goal:** Users can derive formula-backed SOLAR evidence for high-confidence families whose dimensions and memory behavior are visible from reference or workload structure.
**Verified:** 2026-05-23T07:27:20Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SOLAR semantic groups carry parseable per-op formula, byte, and bound evidence inside each group. | VERIFIED | `SolarFormulaEvidence`, `SolarByteEvidence`, and `SolarBoundEvidence` are nested in `SolarSemanticGroupEvidence`; evidence is populated from ordered estimates in `src/sol_execbench/core/scoring/solar_derivation.py`. |
| 2 | Malformed formula, byte, and bound evidence payloads are rejected by the strict parser. | VERIFIED | `_group_from_dict()` requires exact nested keys; `_formula_evidence_from_dict()`, `_byte_evidence_from_dict()`, and `_bound_evidence_from_dict()` validate confidence, numeric bounds, bytes, and limiting resources. Parser rejection tests cover malformed nested payloads. |
| 3 | New evidence fields remain internal sidecar data and do not appear in canonical schemas, primary CLI help, trace JSONL, or AMD-native score artifacts. | VERIFIED | Public guardrails assert `formula_evidence`, `byte_evidence`, and `bound_evidence` are absent from canonical `Definition`, `Workload`, `Trace`, CLI help, and AMD score evidence refs while preserving existing AMD SOL bound fields. |
| 4 | Linear projection remains a first-class `linear_projection` semantic family. | VERIFIED | `_estimate_node()` routes `OpFamily.LINEAR_PROJECTION` through `_gemm_estimate()` while preserving `op_family`; sidecar tests assert `family == "linear_projection"`. |
| 5 | Linear projection reuses GEMM-compatible formulas when dimensions are explicit. | VERIFIED | `_gemm_estimate()` emits `gemm_flops` or `batched_gemm_flops`; regression tests cover ordinary and batched linear projection, including the post-review fix. |
| 6 | Linear projection emits group-local formula, byte, and bound evidence without changing score eligibility. | VERIFIED | Linear projection sidecar tests assert formula, dtype-aware bytes, and bound records in the group; public guardrails prove AMD-native score refs stay sidecar-free. |
| 7 | Explicit attention evidence covers Q/K/V projections, QK scores, scale or mask handling, softmax, PV aggregation, and output projection when visible. | VERIFIED | Attention graph annotation promotes QK, scale/mask, softmax, PV, and output projection nodes; sidecar tests assert Q/K/V projection subroles, formula kinds, byte evidence, and bound evidence. |
| 8 | Incomplete attention axes, mask semantics, head dimensions, or attention subroles degrade or unscore deterministically. | VERIFIED | Attention confidence checks require softmax axis, output projection, formula inputs, bytes, and mask semantics; tests cover partial-mask degraded status and dynamic-axis unscored status without fabricated projection subroles. |
| 9 | Convolution evidence covers 1D, 2D, and 3D convolution, grouped/depthwise metadata, stride, padding, dilation, and output spatial dimensions. | VERIFIED | Graph classification records convolution dimensionality, stride, padding, dilation, groups, and `output_spatial`; estimator formula inputs include channels, groups, kernel elements, and output spatial elements. Tests cover conv metadata, conv3d sidecar evidence, grouped/depthwise estimates, and missing-padding degradation. |
| 10 | Embedding, positional, gather, and rotary-like memory-bound structures emit index/table/output-shape/formula/byte/bound evidence when visible and degrade for dynamic or incomplete metadata. | VERIFIED | Graph annotation records lookup index/table/output metadata and positional/rotary subroles; estimates count index bytes plus selected table/output bytes; tests cover embedding, gather, positional add, rotary-like evidence, and dynamic-index degradation. |
| 11 | Newly promoted high-confidence families emit family-specific formula kinds, formula text, formula inputs, and dtype-aware read/write/intermediate/movement/total byte evidence. | VERIFIED | `OperatorWorkEstimate` fields are serialized into group-local formula and byte evidence for attention, convolution, embedding-positional, and linear projection. Parser and family-modeling tests validate formula kinds and byte buckets. |
| 12 | Family estimates convert into per-operation compute bound, memory bound, limiting resource, and SOL-bound evidence. | VERIFIED | `_bound_evidence_for_estimates()` computes AMD SOL-style compute, memory, limiting resource, and max SOL bounds for every estimate; tests assert bound evidence exists and remains sidecar-only. |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/sol_execbench/core/scoring/solar_derivation.py` | Internal group-local formula, byte, and bound evidence dataclasses, parser, attachment, family gates, and subroles | VERIFIED | Exists, substantive, wired through `derive_solar_derivation_evidence()` and `build_solar_derivation_evidence()`. |
| `src/sol_execbench/core/scoring/amd_bound_graph.py` | High-confidence family recognition attributes for attention, convolution, and memory-bound families | VERIFIED | Exists, substantive, wired through `build_bound_graph()` post-processing annotations. |
| `src/sol_execbench/core/scoring/amd_bound_estimates.py` | Family-specific formula, byte, confidence, and degradation estimates | VERIFIED | Exists, substantive, wired through `estimate_bound_work()` and `_estimate_node()`. |
| `tests/sol_execbench/test_solar_derivation_evidence.py` | Parser, round-trip, deterministic ordering, sidecar boundary, linear projection, and candidate-boundary tests | VERIFIED | Substantive automated coverage included in phase gate. |
| `tests/sol_execbench/test_solar_derivation_family_modeling.py` | Positive/degraded family modeling tests for attention, convolution, embedding, positional, gather, and rotary-like cases | VERIFIED | Substantive automated coverage included in phase gate. |
| `tests/sol_execbench/test_amd_bound_graph.py` | Graph metadata tests for attention, convolution, lookup, and memory-bound attributes | VERIFIED | Substantive automated coverage included in phase gate. |
| `tests/sol_execbench/test_amd_bound_estimates.py` | Estimate formula/byte/confidence tests, including post-review regressions | VERIFIED | Substantive automated coverage included in phase gate. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Canonical schema, CLI, trace JSONL, and AMD-native score guardrails | VERIFIED | Substantive automated coverage included in phase gate. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `BoundGraph` family annotations | `OperatorWorkEstimate` | `_estimate_node()` dispatch by `OpFamily` | WIRED | Attention, convolution, embedding-positional, and linear projection all route to family-specific or GEMM-compatible estimate helpers. |
| `OperatorWorkEstimate` | `SolarSemanticGroupEvidence` | `_formula_evidence_for_estimates()`, `_byte_evidence_for_estimates()`, `_bound_evidence_for_estimates()` | WIRED | Formula, byte, and bound evidence is derived from estimates and attached inside each semantic group. |
| Attention graph nodes | Attention semantic group | `subrole` attributes and `_attention_subroles()` | WIRED | Q/K/V projection evidence, QK scores, scale/mask, softmax, PV, and output projection are represented in attention groups. |
| Convolution graph attributes | Convolution estimates and group evidence | `dimensionality`, `stride`, `padding`, `dilation`, `groups`, `output_spatial` | WIRED | Complete metadata produces scored formula-backed evidence; missing metadata degrades. |
| Lookup/memory graph attributes | Embedding-positional estimates and group evidence | index/table/output metadata and selected-element bytes | WIRED | Lookup estimates count index plus selected table/output bytes; positional and rotary-like paths emit memory-bound evidence. |
| SOLAR sidecar evidence | Public surfaces | guardrail tests | WIRED | Canonical schemas, primary CLI, trace JSONL, and AMD-native score evidence refs remain free of SOLAR sidecar fields. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `solar_derivation.py` | `formula_evidence` | `OperatorWorkEstimate.formula_kind`, `formula`, `formula_inputs` | Yes | FLOWING |
| `solar_derivation.py` | `byte_evidence` | `OperatorWorkEstimate.read_bytes`, `write_bytes`, `intermediate_bytes`, `movement_bytes`, `total_bytes`, plus tensor dtype evidence | Yes | FLOWING |
| `solar_derivation.py` | `bound_evidence` | Estimate FLOPs/bytes converted with default AMD hardware model semantics | Yes | FLOWING |
| `amd_bound_estimates.py` | High-confidence family estimates | Bound graph nodes/tensors and family attributes | Yes | FLOWING |
| `amd_bound_graph.py` | Family/subrole attributes | Reference-visible FX/AST graph extraction and tensor metadata | Yes | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full Phase 49 gate | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0` | `110 passed in 1.22s` | PASS |
| Ruff over touched source/tests | `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/solar_derivation.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_public_contract_guardrails.py` | `All checks passed!` | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| n/a | `find scripts -path '*/tests/probe-*.sh' -type f` | No probes found for this phase | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DERIVE-01 | 49-03 | Attention pattern recognition and degradation | SATISFIED | Attention graph annotation, estimates, sidecar subroles, and tests cover Q/K/V, QK, mask/scale, softmax, PV, output projection, partial mask degradation, and dynamic-axis unscored behavior. |
| DERIVE-03 | 49-04 | 1D/2D/3D convolution recognition and metadata | SATISFIED | Convolution classifiers and estimates include dimensionality, stride, padding, dilation, groups, output spatial metadata, grouped/depthwise formula inputs, and degraded missing metadata. |
| DERIVE-05 | 49-04 | Embedding, positional, gather, and rotary-like structures | SATISFIED | Memory-bound graph attributes and estimates include index/table/output metadata, selected bytes, positional/rotary subroles, and dynamic/incomplete degradation. |
| DERIVE-06 | 49-02 | First-class linear projection with GEMM-compatible formulas | SATISFIED | Linear projection keeps `op_family="linear_projection"` and emits GEMM or batched GEMM formula evidence with complete dimensions. |
| MODEL-01 | 49-01 through 49-04 | Formula kind, formula text, and formula input map | SATISFIED | Group-local `SolarFormulaEvidence` is parseable and populated for promoted high-confidence families. |
| MODEL-02 | 49-01 through 49-04 | Dtype-aware read/write/intermediate/movement/total byte evidence | SATISFIED | Group-local `SolarByteEvidence` serializes byte buckets and dtype inputs from estimates and tensor evidence. |
| MODEL-05 | 49-01 through 49-04 | Compute/memory/limiting/SOL-bound evidence | SATISFIED | Group-local `SolarBoundEvidence` serializes compute bound, memory bound, limiting resource, and SOL bound per operation. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/sol_execbench/core/scoring/amd_bound_graph.py` | 421 | `placeholder` | INFO | Legitimate FX placeholder node handling, not a stub. |
| `src/sol_execbench/core/scoring/amd_bound_graph.py` | 1343 | `return {}` | INFO | Intentional empty attribute map for non-convolution calls, not incomplete implementation. |

### Human Verification Required

None. Phase 49 is backend derivation/modeling code with automated parser, estimator, sidecar, guardrail, and regression coverage. No visual, external-service, real-hardware, or manual UAT item was identified.

### Gaps Summary

No blocking gaps found. The phase goal is achieved in code: high-confidence family modeling exists, is wired through graph extraction, estimate generation, sidecar evidence, parser validation, and guardrail tests, and the full automated phase gate passes.

---

_Verified: 2026-05-23T07:27:20Z_
_Verifier: the agent (gsd-verifier)_
