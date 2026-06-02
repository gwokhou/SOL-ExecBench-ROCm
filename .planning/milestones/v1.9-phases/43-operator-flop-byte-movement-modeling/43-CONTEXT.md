# Phase 43: Operator FLOP/Byte/Movement Modeling - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 43 adds auditable operator-level FLOP, byte, and memory-movement
estimates for the common SOL ExecBench operator families listed in the v1.9
roadmap. The implementation consumes the Phase 42 `BoundGraph` IR and produces
rich estimate evidence with formula metadata, node-local tensor byte accounting,
confidence, and rationale.

This phase corresponds to the formula/evidence conversion step downstream of
the SOLAR graph extractor. It must prepare Phase 44 artifact v2 serialization
and Phase 45 score integration, but it must not implement v2 sidecars, score
report integration, dataset flags, or primary CLI/schema changes.

</domain>

<decisions>
## Implementation Decisions

### Formula Evidence Structure

- **D-01:** Add a new rich operator estimate type for Phase 43 evidence. The
  legacy `amd_sol.WorkEstimate` remains a compatibility view derived from the
  rich estimate.
- **D-02:** Rich estimates must include both machine-checkable formula metadata
  and a human-readable formula string, including fields equivalent to
  `formula_kind`, `formula`, and `formula_inputs`.
- **D-03:** Each `BoundGraphNode` produces one primary rich estimate. Complex
  operator internals are expressed through formula fields, byte buckets,
  confidence, and rationale rather than separate sub-estimates.
- **D-04:** Phase 43 must add `estimate_bound_work(graph)` as the primary API
  consuming `BoundGraph`. Legacy `amd_sol.estimate_work()` remains a
  compatibility adapter returning v1 `WorkEstimate` objects.

### Byte Accounting Semantics

- **D-05:** Rich estimates must expose `read_bytes`, `write_bytes`,
  `intermediate_bytes`, `movement_bytes`, and `total_bytes`, with
  `total_bytes = read_bytes + write_bytes + intermediate_bytes + movement_bytes`.
- **D-06:** Legacy `WorkEstimate.bytes_accessed` is derived from rich
  `total_bytes`.
- **D-07:** Read/write bytes are computed from node-local `BoundTensor` metadata
  referenced by `BoundGraphNode.input_tensor_ids` and `output_tensor_ids`.
- **D-08:** Logical view and broadcast operations keep explicit evidence but
  have zero movement bytes by default.
- **D-09:** `contiguous`, dtype conversion, and other detectable materialization
  paths count movement traffic.
- **D-10:** Missing shape or dtype for a specific byte bucket sets that bucket
  to zero, downgrades confidence, and names the missing evidence in the
  rationale. If all key tensors for an operation are unresolved, the estimate is
  unsupported.

### Operator Coverage Depth

- **D-11:** Phase 43 strictly covers the roadmap success-criteria families:
  GEMM/linear projection, dense matmul, batched matmul, `@`/`torch.mm`/
  `torch.matmul`, elementwise arithmetic, activation chains, reductions,
  normalization/RMSNorm/layer-norm-like patterns, softmax/log-softmax-like
  patterns, data movement, broadcast, contiguous, and dtype conversion.
- **D-12:** Attention, MoE, SSM/Mamba, convolution, embedding/positional, and
  other families outside the Phase 43 roadmap remain explicit unsupported
  estimates. Do not add rough inexact formulas for those families in this
  phase.
- **D-13:** Complex supported/inexact families use conservative pass-count FLOP
  formulas. Reduction, normalization, and softmax estimates must label
  conservative formulas as `inexact` and avoid claiming exact kernel lower
  bounds.
- **D-14:** Elementwise arithmetic and activation chains are estimated per graph
  node. Phase 43 does not infer fusion regions or merge chains into one
  estimate.

### Axis And Shape Evidence

- **D-15:** GEMM and batched GEMM formula inputs such as `M`, `N`, `K`, and
  batch dimensions are inferred from input/output `BoundTensor.shape` metadata.
  Operator/source identity corroborates the interpretation but is not the
  primary source of formula dimensions.
- **D-16:** Reduction, softmax, and normalization estimates prefer node
  attributes such as `dim` or `axis`. When axis metadata is missing, use a
  conservative all-elements estimate, mark confidence `inexact`, and record
  `axis_source="missing"` or equivalent evidence.
- **D-17:** Phase 43 may make small, targeted additions to Phase 42 extractor
  metadata when needed for estimator evidence, such as `dim`, target dtype,
  contiguous/materialization markers, expand/broadcast markers, and movement
  kind. It must not rewrite the extractor architecture or change the public
  graph contract.

### Confidence And Unsupported Semantics

- **D-18:** Known operator semantics with incomplete evidence are `inexact`.
  Unknown operator semantics, unknown tensor effects, unsupported op families,
  unknown dtype byte widths, or inability to resolve all key tensors are
  `unsupported`.
- **D-19:** Unsupported graph nodes still receive one primary estimate with
  zero FLOPs and zero bytes, but they must carry `confidence=unsupported`, an
  explicit warning or rationale, and enough evidence for later phases to avoid
  hidden scoring.
- **D-20:** Missing or unsupported evidence must never be silently treated as
  zero-cost supported work.

### Paper Alignment Boundary

- **D-21:** Phase 43 should stay aligned with the original SOL/SOLAR pipeline:
  Graph Extractor already exists in Phase 42; Phase 43 adds formula/evidence
  conversion; Phase 44/45 handle artifacts and score consumption.
- **D-22:** Do not introduce a standalone `EinsumGraph` or formal extended
  einsum IR in Phase 43. Use SOLAR/einsum-style fields in the rich estimate
  evidence instead.
- **D-23:** Do not fall back to only the legacy roofline-style `WorkEstimate`
  model. The new evidence must be structured and auditable enough for Phase 44
  bound artifact v2.

### Compatibility And Test Surface

- **D-24:** Tests should primarily target `estimate_bound_work(BoundGraph)` and
  rich estimates.
- **D-25:** Add focused compatibility tests for legacy `amd_sol.estimate_work()`
  and v1 bound artifact behavior so existing callers keep working.
- **D-26:** Use golden fixtures with precise assertions for formulas, formula
  inputs, read/write/intermediate/movement/total bytes, confidence, rationale
  snippets, and compatibility downgrade behavior.
- **D-27:** Representative fixtures should cover matmul, batched matmul,
  elementwise + activation chains, reduction, normalization/RMSNorm or
  layer-norm-like patterns, softmax/log-softmax-like patterns, logical views,
  `contiguous`, broadcast/expand, dtype conversion, and unsupported operations.

### the agent's Discretion

The agent may choose exact dataclass names, helper names, formula-kind strings,
test fixture file organization, and whether rich estimates live in a new module
or beside `amd_bound_graph.py`, as long as the decisions above hold and public
benchmark schemas, primary CLI behavior, canonical `Trace` JSONL, and the Phase
42 graph contract remain unchanged.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Original Paper And SOLAR Semantics

- `https://arxiv.org/abs/2603.19173` — Original SOL ExecBench paper. Use as
  the baseline for SOLAR-style graph extraction, formula conversion, SOL bound
  semantics, and no overclaiming.

### Milestone And Phase Scope

- `.planning/PROJECT.md` — v1.9 scope, RDNA 4 validation boundary, deferred
  MI300X-on-CDNA3 and CDNA 4 validation, and paper-alignment constraints.
- `.planning/REQUIREMENTS.md` — Phase 43 MODEL-01 through MODEL-05 and
  downstream BOUND/SCORE/DOC/VAL constraints.
- `.planning/ROADMAP.md` — Phase 43 goal, success criteria, and boundaries
  from Phase 44/45/46.
- `.planning/phases/42-structured-bound-graph-ir/42-CONTEXT.md` — Prior
  decisions that define `BoundGraph`, unsupported/inexact semantics, and the
  compatibility facade boundary.

### Existing Code And Tests

- `src/sol_execbench/core/scoring/amd_bound_graph.py` — Existing structured
  bound graph IR, node/tensor metadata, operation taxonomy, and extractor path.
- `src/sol_execbench/core/scoring/amd_sol.py` — Legacy compatibility facade,
  v1 `WorkEstimate`, v1 bound artifact, and score-bound calculation adapter.
- `src/sol_execbench/core/scoring/__init__.py` — Public scoring exports that
  must remain deliberate.
- `tests/sol_execbench/test_amd_bound_graph.py` — Current graph IR fixtures and
  extractor behavior that Phase 43 can extend with metadata.
- `tests/sol_execbench/test_amd_sol_bounds.py` — Existing v1 estimate and bound
  artifact compatibility tests.
- `tests/sol_execbench/test_public_contract_guardrails.py` — Guardrails proving
  canonical schemas, Trace JSONL, and primary CLI stay unchanged.

### Codebase Maps

- `.planning/codebase/ARCHITECTURE.md` — Layered scoring/reporting architecture
  and public schema stability constraints.
- `.planning/codebase/STACK.md` — Runtime and dependency baseline; do not add a
  heavy graph/symbolic framework dependency for Phase 43.
- `.planning/codebase/CONVENTIONS.md` — Dataclass, module, export, test, and
  error handling conventions.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `BoundGraph`, `BoundGraphNode`, `BoundTensor`, `BoundEdge`, `BoundTensorRole`,
  and `OpFamily` in `src/sol_execbench/core/scoring/amd_bound_graph.py` provide
  the Phase 43 input contract.
- `EstimateConfidence` in
  `src/sol_execbench/core/scoring/amd_hardware_models.py` provides the
  confidence vocabulary for supported, inexact, and unsupported estimates.
- `amd_sol.WorkEstimate`, `OpSolBound`, and `AmdSolBoundArtifact` provide the
  v1 compatibility serialization style that must continue to work.
- `Definition.get_input_shapes()`, `Definition.get_output_shapes()`, and
  concrete `Workload` axis resolution are already used by graph extraction and
  should remain the source of concrete tensor metadata.

### Established Patterns

- Scoring and bound evidence are derived artifacts. They must not mutate
  canonical `Trace` JSONL or public `Definition`, `Workload`, `Solution`, and
  `Trace` schemas.
- Scoring modules use small frozen dataclasses with `to_dict()` serializers and
  focused pytest coverage.
- New public APIs should be deliberately exported from
  `src/sol_execbench/core/scoring/__init__.py`.
- Unsupported and ambiguous evidence is preserved as explicit graph or estimate
  evidence, not dropped.

### Integration Points

- Add the rich estimate API near the scoring modules and expose it through
  `sol_execbench.core.scoring` only if it is intended as public scoring surface.
- Update `amd_sol.estimate_work()` to adapt rich estimates into v1
  `WorkEstimate` without changing the old return type.
- `build_amd_sol_bound_artifact()` can keep emitting the v1 artifact in Phase
  43 by using the compatibility estimate view. Artifact v2 remains Phase 44.
- Phase 43 may extend `amd_bound_graph.py` attributes in a targeted way when
  estimator evidence needs axis, target dtype, broadcast, contiguous, or
  movement-kind metadata.

</code_context>

<specifics>
## Specific Ideas

- Keep the original paper alignment explicit: Phase 43 is formula/evidence
  conversion over a graph, not a score report and not a complete SOL Analyzer.
- A practical API shape is `estimate_bound_work(graph: BoundGraph) ->
  tuple[RichEstimate, ...]`, with each estimate keyed by `node_id`.
- A practical estimate shape includes node ID, op family/name, formula kind,
  formula string, formula inputs, FLOPs, read/write/intermediate/movement/total
  bytes, movement kind or axis source where relevant, confidence, and rationale.
- Golden tests should assert exact bytes and formulas for small concrete
  workloads so later Phase 44/45 agents can rely on deterministic evidence.

</specifics>

<deferred>
## Deferred Ideas

- Full artifact v2 sidecar serialization, loading, coverage summaries, and
  aggregate bound semantics remain Phase 44.
- AMD score report and dataset sidecar integration remain Phase 45.
- User-facing docs, claim guardrails, and RDNA 4 validation evidence remain
  Phase 46.
- Attention, MoE, SSM/Mamba, convolution, embedding/positional, and broader
  paper-scale operator coverage remain future work unless later phases add
  explicit scope.
- A standalone extended-einsum IR remains deferred until the local rich
  estimate and artifact contracts prove stable.

</deferred>

---

*Phase: 43-Operator FLOP/Byte/Movement Modeling*
*Context gathered: 2026-05-23*
