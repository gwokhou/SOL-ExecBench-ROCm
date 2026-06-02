# Phase 44: Bound Artifact V2 And Coverage Semantics - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 44 turns the Phase 42 bound graph and Phase 43 rich operator estimates
into stable AMD SOL bound artifact v2 sidecars with deterministic operation
bounds, aggregate semantics, warnings, and coverage summaries.

This phase delivers a derived scoring artifact and loader/serializer contract.
It does not change canonical `Definition`, `Workload`, `Trace`, `Solution`, or
primary `sol-execbench` CLI behavior. It does not replace the legacy v1
`AmdSolBoundArtifact`; v1 remains a compatibility surface.

</domain>

<decisions>
## Implementation Decisions

### Artifact Contract
- Add a new v2 AMD SOL bound artifact sidecar rather than mutating v1 artifact
  fields.
- The v2 payload should include schema version, derived marker, definition,
  workload UUID, hardware model reference, hardware model payload or evidence
  reference, bound graph payload, rich operator estimates, per-op SOL bounds,
  aggregate bound, warnings, and coverage summary.
- Use existing dataclass + `to_dict()` patterns from scoring modules; avoid a
  new graph or schema framework dependency.
- Provide load/from-dict validation for v2 sidecars so serialized artifacts can
  round-trip and fail clearly on invalid schema versions or missing required
  evidence.

### Bound And Aggregate Semantics
- Per-operation bounds should be computed from rich `OperatorWorkEstimate`
  evidence and the selected AMD hardware model.
- Compute bound, memory bound, limiting resource, confidence, and rationale
  must be recorded for every operation estimate.
- Aggregate bound should not silently improve when evidence is unsupported or
  missing. Unsupported evidence must produce deterministic warnings and an
  unscored/degraded aggregate state rather than hiding zero-cost work.
- Preserve paper-aligned conservative semantics: complete supported evidence
  may score normally, inexact evidence remains visible, and unsupported evidence
  carries explicit degradation.

### Coverage And Warnings
- Coverage summary should report total operations, supported, inexact, and
  unsupported counts by operation family, plus worst confidence.
- Artifact warnings should be deterministic strings derived from graph warnings,
  rich estimate warnings, unsupported/inexact confidence, missing bound evidence,
  and hardware model validation status.
- Warning behavior should be golden-tested so callers can rely on stable
  degradation categories.

### Compatibility And Public Contract
- Keep v1 `AmdSolBoundArtifact`, `WorkEstimate`, and `build_amd_sol_bound_artifact()`
  compatible.
- Add new v2 APIs deliberately through scoring modules; avoid implicit primary
  CLI changes in this phase.
- Existing v1 adapter should continue to derive `bytes_accessed` from rich
  `total_bytes`; v2 should expose the full rich evidence directly.
- Public contract guardrails must continue to prove canonical schemas and
  primary CLI help do not expose v2 sidecar-only fields.

### the agent's Discretion
Implementation details such as exact class names, helper function split, and
file placement are at the agent's discretion as long as they follow the local
scoring module patterns, keep v1 compatibility intact, and satisfy BOUND-01
through BOUND-04.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/scoring/amd_bound_graph.py` provides `BoundGraph`,
  `BoundGraphNode`, tensor metadata, warnings, and JSON-safe `to_dict()`.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py` provides
  `OperatorWorkEstimate`, formula fields, byte buckets, movement evidence,
  confidence, rationale, and warnings.
- `src/sol_execbench/core/scoring/amd_sol.py` provides v1 artifact classes,
  legacy operation bound math, aggregate bound behavior, coverage summary shape,
  and hardware model facade patterns.
- `src/sol_execbench/core/scoring/amd_hardware_models.py` provides strict v2
  hardware model loading and validation statuses.

### Established Patterns
- Scoring artifacts are frozen dataclasses with explicit `to_dict()` methods.
- JSON-safe serialization uses enum `.value`, lists for tuples, and explicit
  dict construction.
- Public exports are deliberate via `src/sol_execbench/core/scoring/__init__.py`.
- Tests live under `tests/sol_execbench/` and focus on schema, guardrail, and
  scoring behavior.

### Integration Points
- New v2 bound artifact APIs should live in scoring modules and consume
  `Definition`, `Workload`, `BoundGraph`, `OperatorWorkEstimate`, and
  `AmdHardwareModel`.
- v1 `build_amd_sol_bound_artifact()` should remain stable and can coexist with
  a new v2 builder.
- Later Phase 45 will consume v2 artifacts from AMD-native scoring and dataset
  reporting, so Phase 44 should expose a clean programmatic API rather than a
  CLI-first workflow.

</code_context>

<specifics>
## Specific Ideas

- Keep alignment with the original SOL/SOLAR paper by modeling this as
  graph/evidence conversion, not as a new unrelated estimator.
- Treat unsupported or missing evidence as visible degradation, never as hidden
  zero-cost work.
- Keep MI300X-on-CDNA3 and CDNA 4 validation claims deferred; any hardware
  status in artifacts must use the Phase 41 split validation fields.

</specifics>

<deferred>
## Deferred Ideas

- Dataset sidecar emission and AMD-native score report consumption belong to
  Phase 45.
- User-facing documentation and RDNA 4 validation closure belong to Phase 46.
- Full original paper 124-model extraction remains future work.

</deferred>
