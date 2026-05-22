# Phase 42: Structured Bound Graph IR - Research

**Researched:** 2026-05-23
**Domain:** SOLAR graph extraction, AMD bound graph IR, PyTorch reference tracing, compatibility facade
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

The user repeatedly selected paper alignment as the controlling decision. Phase
42 therefore plans the SOLAR Graph Extractor layer, not a local AST-only helper
and not the later extended-einsum or SOL Analyzer stages.

### Locked Decisions

- **D-01 to D-04:** The IR is an operator/dataflow graph with explicit node,
  tensor, edge, shape, dtype, source, confidence, and rationale metadata.
- **D-05 to D-08:** Extraction is dynamic-trace-first for a concrete
  `Definition` + `Workload`; static AST remains fallback/source annotation.
- **D-09 to D-10:** Phase 42 stops at the operator graph layer. It may carry
  conversion hints but does not implement a formal extended-einsum graph.
- **D-11 to D-15:** Unsupported or inexact semantics stay visible as graph
  evidence and propagate as coverage debt.
- **D-16 to D-18:** Add a dedicated scoring module and keep `amd_sol.py` as the
  compatibility facade for existing imports.
- **D-19 to D-22:** Keep low-level operator identity and a paper-aligned
  `op_family` taxonomy. Unknown families are explicit unsupported evidence.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
| --- | --- | --- |
| IR-01 | Build a structured AMD SOL bound graph from `Definition` and `Workload` without changing canonical schemas. | Add a new scoring module that consumes existing `Definition.get_input_shapes()` and `Definition.get_output_shapes()` helpers and returns derived dataclasses only. |
| IR-02 | Inspect stable ID, operation family, source expression, tensor roles, resolved shapes, dtypes, confidence, and rationale. | Model `BoundGraph`, `BoundGraphNode`, `BoundTensor`, and edge records with `to_dict()` methods and deterministic IDs. |
| IR-03 | Preserve unsupported or ambiguous constructs as unsupported/inexact evidence. | Dynamic trace failures and static fallback gaps should create graph evidence instead of silently dropping calls. |
| IR-04 | Existing callers of `build_amd_sol_bound_artifact()` and scoring imports continue through a compatibility facade. | Keep `GraphNode`, `extract_graph()`, `estimate_work()`, and `build_amd_sol_bound_artifact()` behavior available in `amd_sol.py`. |

</phase_requirements>

## Summary

The original paper describes SOLAR as three stages: Graph Extractor, Agentic
Einsum Converter, and SOL Analyzer. The Graph Extractor traces a PyTorch model
for a concrete input shape and produces an operator graph with dataflow,
operator types, and intermediate tensor shapes; the later stages convert that
graph to extended einsum form and compute roofline/Orojenesis-style bounds.
[CITED: https://ar5iv.labs.arxiv.org/html/2603.19173v1 Section 4.2]

For this ROCm port, Phase 42 should implement only the first stage. The current
code in `src/sol_execbench/core/scoring/amd_sol.py` is an AST visitor that
returns shallow `GraphNode` records with `node_id`, `op_type`, `expression`,
`confidence`, and `rationale`. That is useful as a compatibility layer, but it
does not carry workload-bound tensor metadata, producer/consumer edges, tensor
roles, intermediate tensors, or paper-aligned operation families.

The practical implementation path is:

1. Add a dedicated module such as
   `src/sol_execbench/core/scoring/amd_bound_graph.py`.
2. Define immutable dataclasses for the graph contract and taxonomy.
3. Implement a `build_bound_graph(definition, workload)` entry point that
   resolves workload shapes and dtypes, runs an isolated reference tracing path
   when possible, and falls back to AST-derived evidence when tracing cannot
   model a construct.
4. Adapt `amd_sol.extract_graph()` to delegate to the new graph builder and
   flatten `BoundGraphNode` records back into the existing `GraphNode` facade.
5. Add golden tests for the paper example family: matmul/projection,
   transpose/data movement, residual add, aliases, tensor methods, chained
   expressions, tuple outputs, and unsupported constructs.

No new runtime dependency is needed. PyTorch is already a runtime dependency,
and the repo already uses stdlib dataclasses plus pytest for scoring contracts.
[VERIFIED: pyproject.toml]

## Paper Alignment Notes

- The paper states that SOLAR derives SOL bounds from PyTorch reference
  implementations and input shapes, not from candidate kernels.
- The Graph Extractor is the paper layer that this phase maps to. It should
  capture eager execution and dynamic control flow when possible.
- The Agentic Einsum Converter and SOL Analyzer are downstream boundaries.
  Phase 42 should leave conversion status/hints only as metadata.
- The paper's illustrative graph contains matmul, transpose, and add nodes with
  intermediate tensor shapes. That should become a golden test target.
- SOLAR's stated limitation that analysis is shape-based, not value-based,
  should appear as inexact/unsupported rationale where a value-dependent
  optimization would be required.

## Existing Code Findings

| Area | Current State | Planning Implication |
| --- | --- | --- |
| `amd_sol.GraphNode` | Shallow AST-derived compatibility record. | Preserve it but do not expand it into the full IR contract. |
| `amd_sol.extract_graph(definition)` | Parses `definition.reference` only. | Keep signature as facade; add new workload-aware graph builder separately. |
| `Definition` helpers | `get_resolved_axes_values`, `get_input_shapes`, `get_output_shapes`. | Reuse for input/output `BoundTensor` metadata. |
| Tests | `test_amd_sol_bounds.py` covers AST extraction, estimates, artifacts, and trace immutability. | Extend tests rather than replacing guardrails. |
| Public contracts | Guardrail tests reject schema/CLI/Trace drift. | Phase 42 must not edit data schemas or primary CLI behavior. |

## Recommended Module Shape

```text
src/sol_execbench/core/scoring/
├── amd_bound_graph.py
└── amd_sol.py

tests/sol_execbench/
├── test_amd_bound_graph.py
├── test_amd_sol_bounds.py
└── test_public_contract_guardrails.py
```

## Validation Architecture

The phase should validate the IR through CPU-only unit tests. GPU/ROCm
hardware is not required because Phase 42 is derived reference analysis, not
performance measurement.

| Validation Target | Automated Command |
| --- | --- |
| New graph contract and extraction tests | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py -x` |
| Compatibility facade and artifact behavior | `uv run pytest tests/sol_execbench/test_amd_sol_bounds.py -x` |
| Public contract guardrails | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -x` |
| Focused phase suite | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` |

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Dynamic tracing executes untrusted or complex reference code during scoring. | Keep the new tracer isolated from benchmark timing/correctness paths, document it as derived analysis, and fail infrastructure errors clearly. |
| Monkeypatch-based tracing misses tensor methods or aliases. | Add AST fallback evidence, preserve unsupported nodes, and cover aliases/tensor methods/chained expressions in tests. |
| Expanding `amd_sol.py` creates a large mixed-responsibility module. | Put IR contracts and extraction in `amd_bound_graph.py`; keep `amd_sol.py` as facade. |
| Phase 42 accidentally implements formula or artifact v2 work. | Keep work estimates and artifact v2 changes in Phase 43/44 plans. |
| Unsupported operations get treated as zero-cost work. | Ensure unsupported/inexact graph evidence survives facade conversion and coverage summaries. |

## Package Legitimacy Audit

No new external packages are recommended or installed in Phase 42. Existing
runtime dependencies already include PyTorch, Pydantic, Click, Rich, and pytest
for tests. The implementation should use stdlib dataclasses, enums, AST, and
the already-present PyTorch runtime only.

## Research Complete

This research supports planning Phase 42 as three executable plans:

1. IR dataclasses, taxonomy, serialization, and metadata contract.
2. Workload-aware dynamic-trace-first extractor with AST fallback evidence.
3. `amd_sol.py` compatibility facade, public exports, and guardrail tests.

