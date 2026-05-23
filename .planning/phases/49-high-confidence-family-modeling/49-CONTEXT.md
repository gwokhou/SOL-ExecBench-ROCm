# Phase 49: High-Confidence Family Modeling - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase promotes structurally visible, high-confidence families from Phase
48 semantic provenance into formula-backed SOLAR derivation evidence. It covers
explicit attention, convolution, embedding/positional/gather/rotary-like memory
patterns, and linear projection when dimensions, dtype, axes, and memory
behavior are visible from the canonical problem definition or workload
structure.

This phase must stay aligned with the original SOL ExecBench paper's SOLAR
semantics: derive formulas from model/problem structure, preserve explicit
degradation for incomplete semantics, and avoid claiming paper-scale dataset or
hardware validation parity. It does not execute submitted candidate solutions
and does not change canonical schemas, primary CLI behavior, score eligibility,
or public benchmark artifacts.

</domain>

<decisions>
## Implementation Decisions

### High-Confidence Family Scope
- Implement only the high-confidence families assigned to Phase 49:
  structurally visible attention, 1D/2D/3D convolution, embedding,
  positional/gather/rotary-like memory-bound structures, and linear projection.
- Attention evidence should cover Q/K/V projections, QK score computation,
  scale or mask handling, softmax, PV aggregation, and output projection when
  axes and mask semantics are visible.
- Convolution evidence should cover dimensionality, grouped or depthwise
  metadata, stride, padding, dilation, and output spatial dimensions.
- Linear projection should become a first-class semantic family while reusing
  GEMM-compatible formula logic when dimensions are explicit.
- MoE, SSM/Mamba, ambiguous attention, and families with dynamic or incomplete
  semantics remain degraded or unsupported and are deferred to Phase 50.

### Formula And Byte Evidence
- Each promoted family must emit a family-specific formula kind, formula text,
  and formula input map.
- Each promoted family must emit dtype-aware read, write, intermediate,
  movement, and total byte evidence.
- Formula and byte evidence must carry provenance through the Phase 48 tensor,
  semantic-axis, source, confidence, missing-evidence, and warning machinery.
- Do not fabricate formulas when required dimensions, dtypes, axes, index
  semantics, mask semantics, grouping metadata, or memory behavior are absent;
  emit deterministic degraded or unsupported evidence instead.

### Extraction Strategy
- Reuse Phase 48 `SolarDerivationEvidence`, semantic groups, source
  boundaries, confidence states, and strict parser/serializer contracts.
- Reuse existing `BoundGraph`, `BoundGraphNode`, `BoundTensor`,
  `OperatorWorkEstimate`, and AMD bound estimate infrastructure where possible
  instead of introducing a separate graph stack.
- Derive evidence from `Definition`, `Workload`, reference-visible FX or AST
  structure, existing shape/dtype resolution, and existing work estimates only.
- Do not read, parse, compile, execute, or infer from submitted candidate
  solution code.
- Avoid new framework dependencies; keep implementation inside the existing
  Python, FX, AST, and scoring stack.

### Integration Boundary
- Phase 49 should produce parseable internal formula-backed evidence usable by
  later sidecar and reporting phases.
- Do not change AMD-native score eligibility, primary `sol-execbench` output,
  canonical trace JSONL, public Pydantic benchmark schemas, or public CLI
  options in this phase.
- Score eligibility, coverage aggregation, and derived report integration are
  deferred to Phase 51.
- Public documentation and claim guardrails for paper parity, hardware parity,
  and hosted leaderboard readiness remain Phase 52 concerns unless local tests
  need targeted guardrails earlier.

### the agent's Discretion
- Exact dataclass names, helper function names, and formula-kind string values
  are at the agent's discretion as long as they are deterministic,
  machine-verifiable, and consistent with existing scoring conventions.
- The planner may split work by family, by shared formula/byte model, or by
  evidence integration layer, provided dependencies remain explicit and tests
  cover each promoted family.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 48 introduced
  `src/sol_execbench/core/scoring/solar_derivation.py` with internal evidence
  dataclasses, strict parser/serializer helpers, semantic grouping, tensor
  provenance, source boundaries, and confidence classification.
- AMD bound graph and estimate code remains in
  `src/sol_execbench/core/scoring/amd_bound_graph.py` and
  `src/sol_execbench/core/scoring/amd_bound_estimates.py`.
- Existing AMD SOL/SOLAR scoring code remains in
  `src/sol_execbench/core/scoring/amd_sol.py`,
  `src/sol_execbench/core/scoring/amd_sol_v2.py`, and
  `src/sol_execbench/core/scoring/amd_score.py`.
- Golden derivation fixture helpers live under
  `tests/sol_execbench/fixtures/solar_derivation/` and
  `tests/sol_execbench/solar_derivation_fixtures.py`.

### Established Patterns
- Internal sidecar artifacts use dataclasses with `to_dict()` methods and
  strict parse helpers.
- Public contract guardrails belong in
  `tests/sol_execbench/test_public_contract_guardrails.py`.
- Focused derivation evidence tests belong in
  `tests/sol_execbench/test_solar_derivation_evidence.py` unless the planner
  creates a more focused family-modeling test file.
- Existing confidence vocabulary is `supported`, `inexact`, and
  `unsupported`, mapped to scored, degraded, and unscored evidence states.

### Integration Points
- Family modeling should extend Phase 48 internal evidence rather than replace
  it.
- Formula-backed family estimates should remain inspectable through sidecar
  evidence, while canonical `Definition`, `Workload`, `Trace`, and primary CLI
  outputs remain unchanged.
- Existing AMD bound graph and v2 sidecar regression tests should remain green
  after Phase 49 changes.

</code_context>

<specifics>
## Specific Ideas

- Treat formulas as inspectable evidence, not hidden implementation details:
  include formula kind, human-readable formula text, concrete inputs, dtype
  bytes, and provenance.
- Prefer conservative supported status: supported only when family, dimensions,
  dtypes, semantic roles, and required metadata are visible.
- Preserve degraded evidence for partially recognized attention masks,
  convolution metadata, gather indices, or rotary-like axis details rather than
  dropping the family entirely.
- Keep Phase 49 tests fixture-driven and local; no ROCm real-hardware
  validation is required for this modeling phase.

</specifics>

<deferred>
## Deferred Ideas

- Conservative MoE and SSM/Mamba modeling remains deferred to Phase 50.
- Sidecar coverage aggregation, AMD-native score eligibility changes, and score
  guard integration remain deferred to Phase 51.
- Dataset-runner reporting closure, public docs, and benchmark claim guardrails
  remain deferred to Phase 52.
- Paper-scale 124-model / 235-problem extraction, MI300X/CDNA validation,
  hosted leaderboard readiness, and NVIDIA Blackwell/B200 equivalence claims
  remain out of scope for v1.10.
</deferred>
