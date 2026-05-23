# Phase 50: Degraded Complex Family Modeling - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase adds conservative derivation paths for complex dynamic families:
MoE and SSM/Mamba-like structures. Unlike Phase 49, the target behavior is not
to maximize scored evidence. The target is to recognize structurally visible
pieces, preserve provenance, and explicitly mark evidence as degraded or
unsupported when static routing, recurrence, cardinality, or state semantics are
incomplete.

This phase must remain aligned with the SOL ExecBench paper's SOLAR derivation
intent: model what is visible from canonical model/problem structure, preserve
clear degradation for incomplete semantic evidence, and avoid paper-scale or
hardware-validation claims. It must not execute submitted candidate solutions,
change canonical benchmark schemas, alter primary CLI behavior, or change
AMD-native score eligibility.

</domain>

<decisions>
## Implementation Decisions

### MoE Scope
- Recognize MoE routing, top-k selection, expert projection, token dispatch,
  and combine patterns when those structures are visible from reference or
  workload evidence.
- Record dynamic routing evidence when expert cardinality, token-to-expert
  assignment, top-k semantics, or static routing cardinality is incomplete.
- Treat incomplete static routing metadata as `inexact` / `degraded`, not as
  scored evidence.
- Use `unsupported` / `unscored` when the pipeline cannot determine core MoE
  semantics or cannot distinguish MoE from unrelated indexing/control-flow
  patterns.

### SSM/Mamba Scope
- Recognize SSM/Mamba-like projection, depthwise convolution, scan or state
  update, gating, and output projection patterns when structurally visible.
- Record recurrence and state-update evidence separately from ordinary
  convolution, data movement, or linear projection evidence.
- Treat incomplete recurrence, scan, state carry, sequence-order, or gating
  semantics as degraded evidence.
- Use `unsupported` / `unscored` when state-update semantics are absent or the
  visible structure is too generic to justify an SSM/Mamba classification.

### Confidence Strategy
- Phase 50 is conservative by default. `supported` is allowed only when family,
  subroles, shapes, dtypes, axes, routing/state semantics, and formula/byte
  provenance are sufficiently visible.
- Common dynamic routing and recurrence cases should become
  `inexact` / `degraded` with explicit `missing_evidence` and stable warning
  prefixes.
- The implementation must never fabricate expert counts, token dispatch
  cardinality, top-k values, sequence state lengths, recurrence semantics, or
  scan behavior.

### Integration Boundary
- Reuse Phase 49 group-local formula, byte, and bound evidence in
  `SolarSemanticGroupEvidence`.
- Reuse existing `BoundGraph`, `OperatorWorkEstimate`, `SolarDerivationEvidence`,
  strict parser, confidence, and public guardrail patterns.
- Do not modify canonical `Definition`, `Workload`, `Trace`, primary
  `sol-execbench` CLI behavior, canonical trace JSONL, or AMD-native score
  eligibility.
- Do not add new framework dependencies and do not execute submitted candidate
  solution code.

### the agent's Discretion
- Exact helper names, formula-kind names, and warning prefixes are at the
  agent's discretion as long as they are deterministic, parseable, and follow
  existing scoring/test conventions.
- The planner may split MoE and SSM/Mamba work by family, by shared degraded
  evidence infrastructure, or by graph/estimate/sidecar layers.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 49 added group-local `SolarFormulaEvidence`, `SolarByteEvidence`, and
  `SolarBoundEvidence` in
  `src/sol_execbench/core/scoring/solar_derivation.py`.
- Family recognition and graph metadata live in
  `src/sol_execbench/core/scoring/amd_bound_graph.py`.
- Formula, byte, confidence, and degradation estimates live in
  `src/sol_execbench/core/scoring/amd_bound_estimates.py`.
- Phase 49 family tests live in
  `tests/sol_execbench/test_solar_derivation_family_modeling.py`.
- Public boundary guardrails live in
  `tests/sol_execbench/test_public_contract_guardrails.py`.

### Established Patterns
- Evidence remains internal and sidecar-only.
- Dataclass sidecars must serialize through explicit `to_dict()` methods and
  parse through strict exact-key helpers.
- Confidence vocabulary remains `supported`, `inexact`, and `unsupported`,
  mapped to scored, degraded, and unscored evidence states.
- Tests should cover positive recognition, degraded incomplete metadata, and
  unsupported/negative behavior.

### Integration Points
- MoE and SSM/Mamba evidence should extend existing semantic groups/subroles
  rather than creating public schema fields.
- Estimates should use `OperatorWorkEstimate` and the Phase 49 evidence
  attachment path.
- Existing high-confidence family behavior from Phase 49 must not regress.

</code_context>

<specifics>
## Specific Ideas

- Treat MoE routing/cardinality and SSM recurrence/state semantics as first
  class missing-evidence categories.
- Prefer degraded evidence for partially visible dynamic structures rather than
  silently dropping them.
- Keep formulas conservative and provenance-rich; if a numeric estimate depends
  on unknown dynamic routing or recurrence, mark it degraded or leave formula
  inputs empty instead of guessing.
- Use Phase 47 MoE and SSM/Mamba fixtures as contract anchors where possible,
  but do not execute fixture reference code.

</specifics>

<deferred>
## Deferred Ideas

- Sidecar coverage aggregation, score eligibility changes, and report guard
  integration remain deferred to Phase 51.
- Dataset-runner reporting closure and public documentation remain deferred to
  Phase 52.
- Paper-scale 124-model / 235-problem extraction, MI300X/CDNA validation,
  hosted leaderboard readiness, and NVIDIA Blackwell/B200 equivalence claims
  remain out of scope.
</deferred>
