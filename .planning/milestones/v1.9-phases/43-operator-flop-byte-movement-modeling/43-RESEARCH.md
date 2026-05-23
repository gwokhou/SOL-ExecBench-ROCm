# Phase 43: Operator FLOP/Byte/Movement Modeling - Research

**Date:** 2026-05-23
**Status:** Complete

## Research Question

What needs to be true to plan Phase 43 well: operator-level FLOP, byte, and
movement evidence over the Phase 42 `BoundGraph`, while staying aligned with the
original SOL/SOLAR paper and preserving ROCm port public contracts?

## Source Findings

### Original Paper Baseline

- The SOL ExecBench paper defines the benchmark target as hardware-grounded SOL
  bounds instead of mutable software-baseline speedups. The abstract states that
  SOLAR computes analytically derived bounds from reference implementations and
  that the SOL score measures progress toward this fixed hardware target.
- Section 2.2 frames the method as roofline-style hardware limits refined by
  Orojenesis-style data movement bounds. For this ROCm port, Phase 43 should
  produce auditable local FLOP/byte/movement evidence, not claim full upstream
  NVIDIA B200/Orojenesis equivalence.
- Section 4.2 is the relevant conceptual boundary: Graph Extractor -> Agentic
  Einsum Converter -> SOL Analyzer. Phase 42 implemented the graph extractor
  analog. Phase 43 should implement formula/evidence conversion over the graph,
  while Phase 44/45 consume it for artifacts and scoring.

References:
- `https://arxiv.org/abs/2603.19173`
- `https://ar5iv.labs.arxiv.org/html/2603.19173v1`

### Local Architecture Findings

- `src/sol_execbench/core/scoring/amd_bound_graph.py` already provides
  workload-bound nodes, tensors, edges, operation families, confidence, source
  expressions, and warnings. It is the correct source of truth for Phase 43.
- `src/sol_execbench/core/scoring/amd_sol.py` still owns the v1 compatibility
  surface: `GraphNode`, `WorkEstimate`, `build_amd_sol_bound_artifact()`, and
  `_bound_for_estimate()`. This should become an adapter over rich estimates,
  not the home of the new model.
- `src/sol_execbench/core/scoring/__init__.py` deliberately exports public
  scoring APIs. New exports should be explicit and tested.
- `tests/sol_execbench/test_amd_bound_graph.py` and
  `tests/sol_execbench/test_amd_sol_bounds.py` are the right focused test
  locations. `tests/sol_execbench/test_public_contract_guardrails.py` protects
  canonical schema and CLI non-leakage.

## Implementation Implications

### Rich Estimate Contract

Create a dedicated scoring module, likely
`src/sol_execbench/core/scoring/amd_bound_estimates.py`, to avoid growing
`amd_sol.py`. The main output should be one frozen dataclass per graph node,
with JSON-safe serialization.

Required fields from context decisions:
- `node_id`
- `op_family`
- `op_name`
- `formula_kind`
- `formula`
- `formula_inputs`
- `flops`
- `read_bytes`
- `write_bytes`
- `intermediate_bytes`
- `movement_bytes`
- `total_bytes`
- `confidence`
- `rationale`

Useful evidence fields for later phases:
- `axis_source`
- `movement_kind`
- `warnings`

### Formula Strategy

Use simple, deterministic, auditable formulas:

- GEMM/mm/matmul: `2*M*N*K*batch`.
- Batched GEMM/bmm: same formula with `batch` or `batch_elements` derived from
  leading dimensions.
- Elementwise arithmetic: one operation per output element.
- Activations: conservative one or more operations per output element, marked
  `inexact`.
- Reduction: conservative pass-count over input elements, marked `inexact`.
- Normalization/RMSNorm/layer-norm-like: conservative pass-count over input
  elements and reduced axis when known, marked `inexact`.
- Softmax/log-softmax-like: conservative max/exp/sum/normalize style
  pass-count over input elements, marked `inexact`.
- Data movement: zero FLOPs, explicit view/materialization movement evidence.

Unsupported families should receive one zero-valued unsupported estimate so
later coverage and unscored behavior can see the missing evidence.

### Byte Strategy

Compute bytes from node-local `BoundTensor` metadata:

- `read_bytes`: sum of resolved input tensor bytes.
- `write_bytes`: sum of resolved output tensor bytes.
- `movement_bytes`: zero for logical view/broadcast, nonzero for `contiguous`,
  dtype conversion, or detectable materialization.
- `intermediate_bytes`: only use when a formula explicitly needs conservative
  temporary traffic evidence; otherwise keep zero to avoid fake precision.
- `total_bytes`: sum all buckets.

Unknown shape or dtype should zero only the affected bucket and downgrade
confidence. Unknown dtype width or no resolvable key tensors should become
unsupported.

### Extractor Metadata Needs

Small metadata additions are justified:

- Capture `dim`/`axis` from reduction, softmax, and normalization calls when it
  appears in args or kwargs.
- Capture dtype conversion target for `.to()`, `.float()`, `.half()`,
  `.bfloat16()`, `.double()`, `.int()`, `.long()`, and similar calls.
- Capture movement kind for `view`, `reshape`, `transpose`, `permute`,
  `expand`, `broadcast_to`, and `contiguous`.

Do not change `BoundGraphNode` public fields; use its existing `attributes`
dictionary.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Rich estimate API accidentally changes v1 artifact semantics | Keep `WorkEstimate` return type unchanged and adapt from rich estimates. |
| Pure view operations are overcounted as memory traffic | Add `movement_kind` and golden tests for logical view/broadcast zero movement. |
| Missing metadata creates fake supported evidence | Downgrade to `inexact` or `unsupported` per Phase 43 context decisions. |
| Unsupported operations disappear from later coverage | Generate one zero-valued unsupported estimate per unsupported node. |
| Phase 43 drifts into artifact v2 or score integration | Keep v2 sidecars and score reports deferred to Phase 44/45. |

## Validation Architecture

The phase is unit-testable without ROCm hardware.

Recommended focused commands:

- `uv run pytest tests/sol_execbench/test_amd_bound_estimates.py -x`
- `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x`
- `uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/amd_sol.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py`

Golden fixtures should cover:

- dense matmul via `@`
- `torch.mm` or `torch.matmul`
- `torch.bmm`
- elementwise + activation chain
- reduction with explicit and missing axis
- RMSNorm/layer-norm-like pattern
- softmax/log-softmax-like pattern
- logical view/transpose/reshape
- broadcast/expand
- `contiguous`
- dtype conversion
- unsupported operation

Assertions should include exact formula kind, formula string, formula inputs,
FLOPs, byte buckets, total bytes, confidence, and rationale snippets.

## Research Complete

Phase 43 should be planned as a scoped scoring-model change: create rich
operator estimates over `BoundGraph`, enrich extractor attributes only where
needed, implement formulas and byte buckets for roadmap families, then adapt
legacy `amd_sol` behavior and guard public contracts.
