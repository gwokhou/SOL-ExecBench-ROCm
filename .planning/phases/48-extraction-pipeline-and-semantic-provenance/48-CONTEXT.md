# Phase 48: Extraction Pipeline And Semantic Provenance - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase adds the shared internal derivation pipeline for compound-family
grouping, subrole metadata, tensor shape, dtype, semantic-axis, extraction
source, and deterministic confidence provenance. It must keep canonical trace
JSONL, public benchmark schemas, primary `sol-execbench` behavior, and score
eligibility unchanged.

This phase does not implement full family-specific formulas for attention,
convolution, embedding, linear projection, MoE, or SSM/Mamba. Those modeling
paths are deferred to Phases 49 and 50. This phase provides the evidence model,
serialization/parsing contract, extraction plumbing, and deterministic
confidence rules that later phases consume.

</domain>

<decisions>
## Implementation Decisions

### Evidence Shape And Storage
- Store new derivation evidence only in internal sidecar/scoring artifacts; do
  not add it to canonical trace JSONL, public Pydantic benchmark schemas, or
  primary CLI default output.
- Record tensor shape, dtype, semantic axes, source kind, source detail, and
  confidence rationale for each derivation node.
- Represent compound-family grouping with a new internal semantic
  group/evidence dataclass that can reference existing `BoundGraphNode` and
  estimate data without rewriting the v1.9 `BoundGraph` core.
- Encode missing evidence with structured `missing_evidence` fields plus stable
  warning prefixes so reports remain readable and tests remain machine
  verifiable.

### Extraction Inputs And Boundaries
- Derive evidence from `Definition.reference`, `Workload.axes`,
  `Workload.inputs`, existing shape/dtype resolution, and FX/AST-visible
  structure only. Do not execute candidate solution code for derivation.
- When static information is incomplete, produce `inexact` or `unsupported`
  evidence with explicit missing fields instead of guessing paper-scale values.
- Do not implement family-specific formulas in Phase 48; keep attention,
  convolution, linear projection, embedding, MoE, and SSM/Mamba formula logic
  for Phases 49 and 50.
- Reuse the existing FX-first and AST-fallback extraction pattern, and tag
  provenance sources such as `fx`, `ast`, `workload`, and `definition`.

### Confidence Rules
- Reuse the existing `supported`, `inexact`, and `unsupported` confidence
  vocabulary and map those states to `scored`, `degraded`, and `unscored`.
- Require visible family, subroles, shape, dtype, semantic axes, and
  formula/source provenance before marking evidence as `supported`.
- Mark evidence as `inexact` when family or subroles are visible but axes,
  mask, routing, padding, recurrence, dtype, byte evidence, or related metadata
  is incomplete.
- Mark evidence as `unsupported` when the pipeline cannot determine family or
  core subrole semantics, or when key dimensions or semantic evidence required
  for formulas are absent; unsupported evidence must carry missing evidence.

### Phase 48 Test And Integration Contract
- The minimum deliverable is a parseable, serializable internal provenance
  evidence contract validated against Phase 47 fixtures for fields, states, and
  sidecar-only boundaries.
- Extend public contract guardrails to prove canonical schemas, trace JSONL,
  and primary CLI options do not expose Phase 48 derivation evidence fields.
- Provide internal derivation evidence and parse/serialize helpers only; do not
  change AMD-native score eligibility or promote the evidence into score
  reports until Phase 51.
- Split implementation planning in dependency order: evidence model and
  serialization, extractor plumbing, deterministic confidence rules, then
  fixture-driven guardrails.

### the agent's Discretion
- Exact class, module, and helper names are at the agent's discretion as long
  as they follow existing scoring/test conventions and remain internal.
- The planner may decide whether the initial evidence model lives in a new
  module or near existing AMD SOL v2 sidecar code, provided public schemas and
  primary CLI behavior remain unchanged.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- AMD scoring and SOL/SOLAR code lives under
  `src/sol_execbench/core/scoring/`, especially `amd_bound_graph.py`,
  `amd_bound_estimates.py`, `amd_sol.py`, `amd_sol_v2.py`, and
  `amd_score.py`.
- Existing `BoundGraph`, `BoundGraphNode`, `BoundTensor`, `OpFamily`,
  `OperatorWorkEstimate`, `AmdSolBoundV2Artifact`, and
  `EstimateConfidence` structures already provide extraction, estimate, and
  confidence vocabulary to reuse.
- Phase 47 fixtures and loader live in
  `tests/sol_execbench/fixtures/solar_derivation/` and
  `tests/sol_execbench/solar_derivation_fixtures.py`.

### Established Patterns
- Keep public benchmark contracts under `src/sol_execbench/core/data/`
  unchanged unless a public schema change is explicitly required.
- Use dataclasses with `to_dict()` / parser helpers for internal scoring and
  sidecar artifacts, matching `amd_sol_v2.py` and hardware model helpers.
- Prefer FX-first and AST-fallback extraction patterns already present in
  `amd_bound_graph.py`.
- Use pytest with real JSON serialization/parsing and focused guardrail tests
  for schema/CLI/claim boundaries.

### Integration Points
- New internal derivation evidence should integrate with
  `src/sol_execbench/core/scoring/` and should be exportable from
  `src/sol_execbench/core/scoring/__init__.py` only if later internal callers
  need package-level imports.
- Focused unit and contract tests should live under `tests/sol_execbench/`,
  reusing the Phase 47 fixture matrix where practical.
- Public contract guardrails should extend
  `tests/sol_execbench/test_public_contract_guardrails.py` without adding
  primary CLI options or canonical trace/schema fields.

</code_context>

<specifics>
## Specific Ideas

- Preserve the paper-aligned semantics while staying explicit that this is not
  the original paper-scale 124-model / 235-problem extraction pipeline.
- Treat Phase 48 as the semantic provenance foundation for Phases 49-51, not as
  a family formula implementation phase.
- Keep unsupported and degraded evidence visible rather than silently dropping
  incomplete derivations.

</specifics>

<deferred>
## Deferred Ideas

- Family-specific formula modeling for high-confidence families is deferred to
  Phase 49.
- Conservative MoE and SSM/Mamba modeling is deferred to Phase 50.
- Sidecar coverage aggregation, AMD-native score eligibility changes, and
  score guard integration are deferred to Phase 51.
- Dataset-runner reporting closure and public documentation are deferred to
  Phase 52.
</deferred>
