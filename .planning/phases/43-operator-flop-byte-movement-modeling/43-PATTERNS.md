# Phase 43: Operator FLOP/Byte/Movement Modeling - Patterns

**Date:** 2026-05-23
**Status:** Complete

## Pattern Map

| Planned File | Role | Closest Existing Analog | Pattern To Reuse |
| --- | --- | --- | --- |
| `src/sol_execbench/core/scoring/amd_bound_estimates.py` | New rich estimate contract and estimator | `src/sol_execbench/core/scoring/amd_bound_graph.py` | Frozen dataclasses, enum/string taxonomy, JSON-safe `to_dict()`, explicit warnings. |
| `src/sol_execbench/core/scoring/amd_bound_graph.py` | Targeted extractor metadata enrichment | Existing same file | Keep `BoundGraphNode.attributes` as metadata extension point; do not add schema fields. |
| `src/sol_execbench/core/scoring/amd_sol.py` | Legacy compatibility adapter | Existing same file | Preserve v1 dataclasses and return types; delegate richer logic behind the facade. |
| `src/sol_execbench/core/scoring/__init__.py` | Public scoring exports | Existing same file | Add deliberate exports only for intended public scoring surface. |
| `tests/sol_execbench/test_amd_bound_estimates.py` | Golden estimate tests | `tests/sol_execbench/test_amd_bound_graph.py` and `tests/sol_execbench/test_amd_sol_bounds.py` | Small inline `Definition`/`Workload` fixtures with exact assertions. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Contract non-leakage guardrails | Existing same file | Assert derived artifacts do not enter canonical schemas or primary CLI. |

## Existing Code Excerpts To Follow

- `BoundGraph.to_dict()` sorts tensor payloads and returns plain JSON-like
  values. Rich estimates should follow the same pattern.
- `AmdSolBoundArtifact.to_dict()` serializes dataclass tuples by calling each
  nested object's `to_dict()`. Phase 43 should keep that convention.
- `_dtype_bytes()` in `amd_sol.py` already maps project `DType` values to byte
  widths. Move or mirror this logic carefully so dtype byte widths are shared by
  rich estimates and legacy adapters.
- `_legacy_op_type()` in `amd_sol.py` maps `OpFamily` values back to old
  operation strings. Keep this as the compatibility boundary.

## Integration Notes

- New estimate APIs should consume `BoundGraph`, not raw `Definition` and
  `Workload`, because Phase 42 already resolved workload-specific graph/tensor
  metadata.
- `amd_sol.estimate_work(definition, workload, graph_nodes)` cannot recover all
  new graph metadata from legacy `GraphNode`. The compatibility path should
  build a `BoundGraph` internally when possible, then degrade to the existing
  AST fallback only if needed.
- `build_amd_sol_bound_artifact()` remains v1 in Phase 43. It can use the
  legacy adapted `WorkEstimate` values and should not emit v2 sidecar fields.

## Test Fixture Patterns

- Use concrete small shapes so formulas are obvious:
  - matmul: `(M,K) @ (K,N)` with `M=2`, `K=4`, `N=8` -> `2*M*N*K = 128`.
  - batched matmul: `(B,M,K) @ (B,K,N)` with `B=3`, `M=2`, `K=4`, `N=8` -> `384`.
  - elementwise chain: vector length `16`, two nodes, each node estimates its own work.
- Assert both numeric values and evidence strings:
  - `formula_kind`
  - `formula`
  - `formula_inputs`
  - `read_bytes`
  - `write_bytes`
  - `movement_bytes`
  - `total_bytes`
  - `confidence`
  - rationale substring

## Pattern Complete

The plan should create a new rich estimate module, add only targeted graph
attribute metadata, adapt old `amd_sol` APIs, and lock behavior with golden CPU
unit tests.
