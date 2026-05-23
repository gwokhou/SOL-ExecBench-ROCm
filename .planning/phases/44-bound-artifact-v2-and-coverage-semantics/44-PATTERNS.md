# Phase 44: Bound Artifact V2 And Coverage Semantics - Patterns

**Date:** 2026-05-23
**Status:** Complete

## Pattern Map

| Planned File | Role | Closest Existing Analog | Pattern To Reuse |
| --- | --- | --- | --- |
| `src/sol_execbench/core/scoring/amd_sol_v2.py` | New v2 sidecar contract, loader, builder, coverage, warnings | `amd_sol.py`, `baseline_artifact.py`, `amd_hardware_models.py` | Frozen dataclasses, explicit schema constants, strict from-dict validation, JSON-safe `to_dict()`. |
| `src/sol_execbench/core/scoring/__init__.py` | Deliberate public scoring exports | Existing same file | Add explicit imports and `__all__` entries only for intended v2 APIs. |
| `tests/sol_execbench/test_amd_sol_v2.py` | Golden v2 artifact behavior | `test_amd_sol_bounds.py`, `test_amd_bound_estimates.py` | Inline `Definition`/`Workload` fixtures with exact serialized payload assertions. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Compatibility guardrails | Existing same file | Assert derived sidecar-only fields stay out of canonical schemas and primary CLI help. |

## Existing Code Excerpts To Follow

- `AmdHardwareModel.to_dict()` uses `asdict()` but replaces enum values and
  tuples with JSON-safe strings/lists. v2 nested payloads should do the same.
- `BoundGraph.to_dict()` sorts tensor payloads for deterministic output. v2
  should preserve this graph payload rather than rebuild its own graph schema.
- `OperatorWorkEstimate.to_dict()` already exposes formula, bytes, confidence,
  rationale, and warnings. v2 should include it as the estimate evidence
  payload.
- `amd_hardware_model_from_dict()` rejects unknown/missing fields and invalid
  enum values with `ValueError`. The v2 loader should use the same strict,
  actionable failure style.
- `amd_sol._bound_for_estimate()` contains the current compute/memory SOL math.
  v2 can use equivalent math over rich estimate `flops` and `total_bytes`.

## Integration Notes

- Prefer a new module over adding many v2 concerns to `amd_sol.py`; the old
  module is already the v1 compatibility facade.
- The builder should consume `Definition`, `Workload`, `AmdHardwareModel`, and
  optional `hardware_model_ref`; internally it should call `build_bound_graph()`
  and `estimate_bound_work()`.
- Keep v2 APIs programmatic. Dataset sidecar emission and score consumption are
  Phase 45 concerns.
- Avoid new dependencies; local dataclasses and parsing helpers are sufficient.

## Test Fixture Patterns

- Matmul fixture:
  - `(M,K) @ (K,N)` with `M=2`, `K=4`, `N=8`.
  - Expected FLOPs `128`, total bytes `224`, one supported op, scored or
    degraded depending on hardware validation status.
- Elementwise/reduction fixture:
  - multiple inexact operations with known family counts.
  - Expected aggregate `degraded`, worst confidence `inexact`.
- Unsupported fixture:
  - `torch.linalg.inv(x)`.
  - Expected one unsupported op, warning prefix `unsupported_operator:`, and
    aggregate status `unscored`.
- Loader fixture:
  - serialize a valid payload, load it, serialize again, and compare stable
    fields.
  - mutate schema version or remove a required field and assert `ValueError`.

## Pattern Complete

The implementation should add one focused v2 sidecar module, export it
deliberately, cover it with CPU golden tests, and preserve v1 plus canonical
schema behavior through explicit guardrails.
