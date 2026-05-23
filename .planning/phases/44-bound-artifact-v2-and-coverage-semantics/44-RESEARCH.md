# Phase 44: Bound Artifact V2 And Coverage Semantics - Research

**Date:** 2026-05-23
**Status:** Complete

## Research Question

How should the ROCm port turn the Phase 42 `BoundGraph` and Phase 43 rich
operator estimates into a stable AMD SOL bound artifact v2 sidecar while
remaining aligned with the original SOL/SOLAR paper and preserving public
benchmark contracts?

## Source Findings

### Original Paper Baseline

- The SOL ExecBench paper frames SOL/SOLAR as a hardware-grounded analytic
  target: graph extraction and analytic bound generation create fixed
  speed-of-light reference points independent of mutable software baselines.
- Phase 44 maps most directly to the paper's SOL Analyzer artifact boundary:
  it should package graph evidence, per-operation analytic limits, aggregate
  bound semantics, and coverage confidence into an auditable derived artifact.
- This ROCm port should not claim upstream NVIDIA B200/Orojenesis equivalence
  from this phase alone. It should expose the AMD evidence and degradation
  states needed for later scoring/reporting phases.

References:
- `https://arxiv.org/abs/2603.19173`
- `https://ar5iv.labs.arxiv.org/html/2603.19173v1`

### Local Architecture Findings

- `src/sol_execbench/core/scoring/amd_bound_graph.py` already serializes graph
  nodes, tensors, edges, warnings, definition, workload UUID, and derived
  marker. It is the correct graph evidence payload for v2.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py` already provides one
  rich `OperatorWorkEstimate` per graph node, including FLOPs, byte buckets,
  formula evidence, confidence, rationale, and warnings.
- `src/sol_execbench/core/scoring/amd_sol.py` owns v1 compatibility:
  `AmdSolBoundArtifact`, `WorkEstimate`, `OpSolBound`, and
  `build_amd_sol_bound_artifact()`. v2 should coexist with this surface rather
  than mutate it.
- `src/sol_execbench/core/scoring/amd_hardware_models.py` already provides the
  strict v2 hardware model object and split validation status fields needed in
  the sidecar.
- `tests/sol_execbench/test_amd_sol_bounds.py`,
  `tests/sol_execbench/test_amd_bound_estimates.py`, and
  `tests/sol_execbench/test_public_contract_guardrails.py` provide the focused
  CPU verification surface.

## Implementation Implications

### Artifact Contract

Add a dedicated v2 contract, likely in a new
`src/sol_execbench/core/scoring/amd_sol_v2.py` module, to avoid overloading the
v1 compatibility module.

The artifact should include:

- schema version and derived marker
- definition name and workload UUID
- hardware model reference
- hardware model payload
- bound graph payload
- rich operator estimate payloads
- per-operation SOL bounds
- aggregate bound object
- deterministic warnings
- coverage summary

Provide both `to_dict()` and `amd_sol_bound_v2_from_dict()` style loading.
Validation should fail clearly on invalid schema version, missing required
fields, malformed nested lists/dicts, or invalid confidence/status values.

### Bound Semantics

Per-operation bounds should reuse the existing v1 math:

- compute bound in ms = FLOPs / peak TFLOPs
- memory bound in ms = total bytes / memory bandwidth
- SOL bound = max(compute bound, memory bound)
- limiting resource = `compute` or `memory`

The v2 bound should use `OperatorWorkEstimate.total_bytes`, not legacy
`WorkEstimate.bytes_accessed`, and should retain `op_family`, `op_name`,
confidence, rationale, and estimate warnings.

### Aggregate Semantics

Aggregate state should be deterministic and conservative:

- `scored`: all operation estimates are supported and hardware/model validation
  statuses are not unvalidated.
- `degraded`: no unsupported operation evidence, but at least one estimate is
  inexact or hardware/model status is provisional.
- `unscored`: at least one operation estimate is unsupported or required
  bound evidence is missing.

The aggregate object may still include the numeric sum of per-op bounds for
debugging, but callers must inspect the state. Unsupported evidence must emit
warnings so a zero-valued unsupported estimate cannot silently improve scores.

### Coverage And Warnings

Coverage should summarize:

- total operation count
- supported/inexact/unsupported counts
- counts by operation family
- supported/inexact/unsupported counts by operation family
- worst confidence

Warning strings should be stable and machine-checkable. Recommended prefixes:

- `graph_warning:`
- `estimate_warning:`
- `inexact_operator:`
- `unsupported_operator:`
- `hardware_validation:`
- `model_validation:`
- `aggregate_unscored:`
- `aggregate_degraded:`

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| v2 artifact accidentally changes v1 artifact payloads | Implement a new module/API and add regression assertions that v1 lacks v2-only fields. |
| Unsupported operations appear as zero-cost speedups | Mark aggregate `unscored` and emit deterministic unsupported warnings. |
| Inexact evidence loses visibility in aggregate reports | Include coverage worst confidence and degraded aggregate warnings. |
| Loader accepts ambiguous sidecar payloads | Require schema version and all top-level fields; reject missing or malformed fields. |
| New API leaks into primary CLI or canonical schemas | Keep Phase 44 programmatic only and extend public contract guardrails. |

## Validation Architecture

This phase is CPU-unit-testable; no ROCm hardware is required.

Recommended commands:

- `uv run pytest tests/sol_execbench/test_amd_sol_v2.py -x`
- `uv run pytest tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x`
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_sol_v2.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_public_contract_guardrails.py`

Golden fixtures should cover:

- supported matmul artifact round-trip
- inexact elementwise/reduction coverage and degraded aggregate
- unsupported operation coverage and unscored aggregate
- invalid schema/missing required field loader failures
- v1 payload compatibility and primary CLI non-leakage

## Research Complete

Phase 44 should be planned as a derived scoring artifact change: add a v2
sidecar contract, compute per-op and aggregate bounds from rich estimates,
make coverage/warnings deterministic, export the new API deliberately, and
preserve v1/canonical contract boundaries.
