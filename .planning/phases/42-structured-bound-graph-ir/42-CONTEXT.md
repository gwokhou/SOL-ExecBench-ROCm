# Phase 42: Structured Bound Graph IR - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 42 establishes a paper-aligned structured bound graph IR for the AMD
SOL/SOLAR modeling pipeline. It replaces the current AST-centric operation list
with an inspectable operator/dataflow graph that can carry tensor metadata,
producer/consumer relationships, operation families, source attribution,
confidence, and rationale for a concrete `Definition` + `Workload`.

This phase corresponds to the SOLAR Graph Extractor layer described in the
original paper. It should prepare downstream Phase 43 formula/einsum-style
modeling and Phase 44 artifact v2 serialization, but it must not implement the
full extended-einsum converter, SOL analyzer, dataset integration, score-report
integration, or new public CLI/schema behavior.

</domain>

<decisions>
## Implementation Decisions

### Paper-Aligned IR Shape

- **D-01:** The Phase 42 IR must be a paper-aligned operator/dataflow graph, not
  only a richer version of the existing AST expression list.
- **D-02:** The graph must represent operator nodes and tensor/value metadata
  explicitly enough to capture dataflow, operation types, intermediate tensor
  shapes, and dtypes for a concrete workload.
- **D-03:** Stable graph node IDs must remain deterministic for the same
  `Definition` + `Workload` extraction path so tests and downstream artifacts
  can reference nodes reliably.
- **D-04:** The IR should expose at least `BoundGraph`, `BoundGraphNode`,
  `BoundTensor`, producer/consumer edges, operation attributes, tensor roles,
  resolved shapes, dtypes, source expressions, confidence, and rationale.

### Extraction Strategy

- **D-05:** Graph extraction should be dynamic-trace-first to align with SOLAR:
  the preferred path executes or instruments the PyTorch reference for a
  concrete workload to capture the actual eager execution path.
- **D-06:** Static AST analysis may remain as a fallback and as a source
  attribution/rationale helper, but it must not be treated as the final IR
  semantic boundary.
- **D-07:** Reference execution for IR extraction must run through an isolated
  tracer path that is separate from benchmark timing, correctness evaluation,
  canonical `Trace` JSONL generation, and primary CLI output.
- **D-08:** If dynamic tracing fails or sees behavior it cannot model, the graph
  must preserve explicit inexact or unsupported evidence instead of silently
  producing a clean-looking supported graph.

### SOLAR Pipeline Boundary

- **D-09:** Phase 42 implements the operator graph layer only. It should be
  designed to feed later extended-einsum/formula conversion, but it does not
  need to expose a formal `EinsumGraph` or compute FLOP/byte formulas.
- **D-10:** The IR may include non-authoritative hints such as `einsum_hint`,
  `index_pattern_hint`, or `conversion_status` when cheap and useful, but these
  are metadata for later phases, not Phase 42's formal converter output.

### Unsupported And Inexact Semantics

- **D-11:** Unsupported or ambiguous Python/PyTorch behavior must remain visible
  as graph evidence. Do not drop unknown calls or tensor effects.
- **D-12:** Use `confidence=inexact` when the extractor knows op, shape, dtype,
  and dataflow but cannot fully guarantee later modeling semantics.
- **D-13:** Use `confidence=unsupported` when operator semantics, tensor effects,
  or conversion readiness cannot be reliably established.
- **D-14:** Graph extraction should fail only for infrastructure-level failures
  such as unusable reference code, impossible input generation, or tracer setup
  errors. Semantic non-coverage should become graph evidence instead.
- **D-15:** Coverage debt must propagate forward. Later bound artifacts and
  scoring must not treat unsupported or missing evidence as zero-cost hidden
  work.

### Module Boundary And Compatibility

- **D-16:** Implement the new IR in a dedicated scoring module rather than
  continuing to grow `src/sol_execbench/core/scoring/amd_sol.py`.
- **D-17:** `amd_sol.py` remains the compatibility facade for existing imports,
  including `extract_graph()`, `estimate_work()`, `build_amd_sol_bound_artifact()`,
  and `GraphNode`-style compatibility behavior.
- **D-18:** Existing callers must continue to work through the facade. Any new
  paper-aligned IR API should be exported deliberately through
  `src/sol_execbench/core/scoring/__init__.py` only when intended as public
  scoring surface.

### Operation Taxonomy

- **D-19:** IR nodes should retain both low-level operator identity
  (`op_name`, callable identity, or ATen/PyTorch name when available) and a
  paper-aligned `op_family` for coverage and modeling.
- **D-20:** Initial paper-aligned families should include attention, MoE,
  normalization, embedding/positional, linear/projection, GEMM,
  MLP/activation, convolution, SSM/Mamba, softmax, reduction, elementwise,
  data movement, dtype conversion, and unsupported.
- **D-21:** Current local families should map into the paper-aligned taxonomy
  where possible: matmul to GEMM or linear/projection based on context,
  activation to MLP/activation, plus existing normalization, softmax, reduction,
  elementwise, and data-movement categories.
- **D-22:** Unknown families must be explicit (`unsupported` or equivalent
  non-supported evidence with rationale), never silently omitted.

### the agent's Discretion

The agent may choose exact module names, dataclass names, tracing mechanism,
and compatibility adapter details as long as the decisions above hold and public
benchmark schemas/CLI/Trace JSONL remain unchanged.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Original Paper And SOLAR Semantics

- `https://arxiv.org/abs/2603.19173` — Original SOL-ExecBench paper. Use as
  the baseline for SOLAR pipeline semantics and no overclaiming.
- `https://ar5iv.labs.arxiv.org/html/2603.19173v1` §4.2 — SOLAR bound
  derivation. Key reference for Graph Extractor -> Agentic Einsum Converter ->
  SOL Analyzer boundaries.

### Milestone And Phase Scope

- `.planning/PROJECT.md` — v1.9 scope, RDNA 4 validation boundary, deferred
  CDNA 3 / MI300X and CDNA 4 validation, and paper-parity constraints.
- `.planning/REQUIREMENTS.md` — Phase 42 requirements IR-01 through IR-04 and
  downstream MODEL/BOUND constraints.
- `.planning/ROADMAP.md` — Phase 42 goal, success criteria, and downstream
  phase boundaries for operator modeling and bound artifact v2.
- `.planning/phases/41-bound-model-contract-and-hardware-artifacts/41-CONTEXT.md`
  — Prior decisions about hardware model artifacts, public-contract guardrails,
  and score/report integration deferral.

### Existing Code And Tests

- `src/sol_execbench/core/scoring/amd_sol.py` — Current AST-centric graph,
  work estimate, bound artifact, and compatibility facade to evolve.
- `src/sol_execbench/core/scoring/amd_hardware_models.py` — Phase 41 hardware
  model contract consumed by bound artifact compatibility.
- `src/sol_execbench/core/scoring/__init__.py` — Public scoring exports that
  must remain deliberate.
- `tests/sol_execbench/test_amd_sol_bounds.py` — Existing behavior and public
  compatibility tests for graph extraction, estimates, unsupported operations,
  trace immutability, and documentation guardrails.
- `tests/sol_execbench/test_public_contract_guardrails.py` — Guardrails proving
  public schemas, primary CLI help, and canonical Trace JSONL stay unchanged.

### Codebase Maps

- `.planning/codebase/ARCHITECTURE.md` — Layered scoring/reporting architecture
  and public schema stability constraints.
- `.planning/codebase/STACK.md` — Runtime and dependency baseline; no new heavy
  graph/symbolic framework dependency is implied by Phase 42.
- `.planning/codebase/CONVENTIONS.md` — Dataclass, module, export, test, and
  error handling conventions.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/sol_execbench/core/scoring/amd_sol.py` already exposes the compatibility
  surface that downstream callers use: `extract_graph()`, `estimate_work()`,
  `summarize_amd_sol_coverage()`, and `build_amd_sol_bound_artifact()`.
- Existing dataclasses `GraphNode`, `WorkEstimate`, `OpSolBound`, and
  `AmdSolBoundArtifact` provide a local serialization style, but the current
  `GraphNode` is too shallow for paper-aligned SOLAR graph extraction.
- `Definition.get_resolved_axes_values()`, `Definition.get_input_shapes()`, and
  `Definition.get_output_shapes()` already provide shape resolution for a
  concrete workload and should be reused by the tracer/IR builder.
- Phase 41's `AmdHardwareModel` loader and packaged `gfx1200` model should
  remain the hardware-model source for bound artifacts built through the
  compatibility facade.

### Established Patterns

- Scoring and bound artifacts are derived evidence. They must not mutate
  canonical `Trace` JSONL or public `Definition`, `Workload`, `Solution`, and
  `Trace` schemas.
- Public schema models live under `src/sol_execbench/core/data/` and are guarded
  by tests. Phase 42 should not add fields there.
- Tests favor focused unit coverage for scoring contracts plus public-contract
  guardrails when derived artifacts are introduced.
- Package modules under `src/sol_execbench/core/scoring/` use small immutable
  dataclasses with `to_dict()` serialization methods.

### Integration Points

- `amd_sol.extract_graph()` should adapt or delegate to the new IR extractor
  while preserving the existing return shape expected by tests and callers.
- `amd_sol.build_amd_sol_bound_artifact()` should continue to work, either by
  consuming the new IR directly or by adapting the IR into the existing v1 bound
  artifact shape until Phase 44 introduces artifact v2.
- `amd_score.py` should not gain full v2 score-report integration in Phase 42;
  that remains Phase 45 scope.

</code_context>

<specifics>
## Specific Ideas

- The implementation should mirror the original SOLAR separation:
  Graph Extractor now, extended-einsum/formula conversion later, SOL Analyzer
  and artifact v2 later.
- A practical Phase 42 target is a scoped dynamic tracer for common PyTorch
  reference patterns, backed by AST/source annotations and explicit fallback
  evidence when tracing cannot model something.
- The graph should make later formulas possible without re-parsing raw Python:
  operation attributes, tensor role metadata, shape/dtype evidence, and
  producer/consumer edges are required.
- The paper's examples around matmul/projection plus residual add should guide
  initial golden tests: a graph with matmul/projection, add/elementwise, and
  transpose/data-movement-like nodes plus intermediate tensor shapes.

</specifics>

<deferred>
## Deferred Ideas

- Full agentic extended-einsum conversion with LLM-generated conversion
  functions and self-correction remains outside Phase 42.
- Full SOL Analyzer implementation, Orojenesis-style tighter data movement
  modeling, and aggregate bound semantics remain Phase 43/44 or future scope.
- Dataset runner flags, AMD score report integration, and optional sidecar
  emission remain Phase 45 scope.
- CDNA 3 / MI300X real-hardware validation and CDNA 4 validation remain
  deferred from v1.9.

</deferred>

---

*Phase: 42-Structured Bound Graph IR*
*Context gathered: 2026-05-23*
