# Phase 41: Bound Model Contract And Hardware Artifacts - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 41 establishes the artifact and hardware-model contract foundation for
v1.9 before the IR and operator estimator work begins. It should define where
AMD hardware model JSON artifacts live, how they are loaded and validated, how
v2 validation status fields replace the old single `validation_status`, and
which public-contract/claim guardrails must be in place early.

This phase does not implement the structured IR, operator FLOP/byte formulas,
dataset integration flags, or score-report integration beyond the contract and
guardrails needed to support later phases.

</domain>

<decisions>
## Implementation Decisions

### Artifact Location And Packaging

- **D-01:** Default AMD hardware model JSON artifacts must be packaged with the
  Python package under `src/sol_execbench/data/amd_hardware_models/`, starting
  with an RDNA 4 `gfx1200` artifact.
- **D-02:** Packaged artifacts should be loaded through Python package resource
  APIs such as `importlib.resources`, so defaults remain available after
  package installation.
- **D-03:** Phase 41 should implement loader support for an arbitrary external
  hardware model JSON path, but should not add dataset-runner or CLI flags yet.
  Wiring external paths into `scripts/run_dataset.py` remains Phase 45 scope.

### Hardware Validation Status Semantics

- **D-04:** v2 hardware model artifacts must distinguish hardware/environment
  validation from model validation with two fields:
  `hardware_validation_status` and `model_validation_status`.
- **D-05:** RDNA 4 hardware validation context may be represented separately
  from model validation, but the Phase 41 bound model itself remains
  provisional until Phase 46 records bound-model validation evidence.
- **D-06:** v2 artifacts hard-replace the old single `validation_status` field.
  `validation_status` is not part of the v2 contract. Legacy v1 compatibility
  code may translate from the old field internally, but v2 JSON must use the two
  explicit status fields.

### Artifact Schema Strictness

- **D-07:** v2 hardware model and bound artifact loaders must reject unknown
  fields instead of silently accepting schema drift.
- **D-08:** Invalid packaged artifacts should fail explicitly during loading or
  tests; they must not fall back to hard-coded model constants.

### Fallback Model Policy

- **D-09:** Hard-coded peak compute, memory bandwidth, source, and validation
  metadata must be moved out of `default_amd_hardware_models()` and into
  packaged JSON artifacts.
- **D-10:** `default_amd_hardware_models()` may remain as a compatibility API,
  but it must load packaged JSON artifacts. If packaged JSON is missing or
  invalid, it should fail explicitly rather than returning embedded constants.

### Public-Contract Guardrail Depth

- **D-11:** Phase 41 must guard canonical `Trace` JSONL, primary
  `sol-execbench` CLI behavior, and public definition/workload/solution schemas
  against accidental bound-modeling changes.
- **D-12:** Phase 41 must also add documentation or grep-style guardrails that
  block premature B200, upstream SOLAR, leaderboard-equivalence, CDNA 3 /
  MI300X validation, and CDNA 4 validation claims in v1.9 docs or outputs.
- **D-13:** Full score-report warning integration remains Phase 45 scope unless
  a small compatibility shim is required to keep existing tests passing.

### the agent's Discretion

The agent may choose exact module names, dataclass/Pydantic split, and test file
organization as long as the decisions above are preserved, existing public
imports are handled deliberately, and canonical trace/CLI contracts do not
change.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone And Phase Scope

- `.planning/PROJECT.md` — Project constraints, v1.9 scope, RDNA 4 validation
  boundary, and deferred CDNA 3 / MI300X and CDNA 4 validation.
- `.planning/REQUIREMENTS.md` — Phase 41 requirements HW-01 through HW-04 and
  DOC-01, plus out-of-scope items.
- `.planning/ROADMAP.md` — Phase 41 goal, success criteria, and downstream
  phase boundaries.
- `.planning/research/SUMMARY.md` — v1.9 research synthesis, recommended
  artifact flow, guardrails, and module split.

### Codebase Patterns

- `.planning/codebase/ARCHITECTURE.md` — Layered architecture and scoring as a
  derived subsystem.
- `.planning/codebase/STACK.md` — Existing dependency stack; no new graph or
  symbolic math framework dependency is expected.
- `.planning/codebase/CONVENTIONS.md` — Python module, dataclass, Pydantic,
  error handling, and test naming conventions.

### Source And Tests

- `src/sol_execbench/core/scoring/amd_sol.py` — Current `AmdHardwareModel`,
  bound artifact v1, `default_amd_hardware_models()`, and compatibility surface
  that Phase 41 must migrate carefully.
- `src/sol_execbench/core/scoring/amd_score.py` — Derived AMD score warning and
  evidence-reference behavior that later phases consume.
- `tests/sol_execbench/test_amd_sol_bounds.py` — Existing AMD SOL artifact and
  trace immutability tests that Phase 41 should evolve.
- `tests/sol_execbench/test_amd_native_score.py` — Existing score evidence and
  hardware warning tests.
- `tests/sol_execbench/test_public_contract_guardrails.py` — Existing public
  schema, CLI, trace, and claim guardrail tests.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `AmdHardwareModel`, `EstimateConfidence`, and `HardwareValidationStatus` in
  `src/sol_execbench/core/scoring/amd_sol.py` provide the current dataclass and
  enum pattern, but the v2 hardware model contract needs two validation status
  fields instead of one.
- `ScoringBaselineArtifact` loading in
  `src/sol_execbench/core/scoring/baseline_artifact.py` is a useful local
  pattern for JSON artifact parsing and clear validation errors.
- Existing `AmdSolBoundArtifact.to_dict()` and score report `to_dict()` methods
  show the derived artifact serialization style.

### Established Patterns

- Scoring artifacts are derived and separate from canonical `Trace` JSONL.
- Tests protect public schemas and primary CLI behavior before implementation
  work changes internals.
- Package source uses small dataclasses for derived evidence and Pydantic for
  public schemas; Phase 41 should keep that style.

### Integration Points

- `default_amd_hardware_models()` is the compatibility entry point most likely
  to be touched by Phase 41.
- Later Phase 45 will integrate external hardware model paths with
  `scripts/run_dataset.py`; Phase 41 should make this possible through loader
  APIs without adding that user-facing option yet.
- Public contract guardrails should live near existing tests in
  `tests/sol_execbench/test_public_contract_guardrails.py` and
  `tests/sol_execbench/test_amd_sol_bounds.py`.

</code_context>

<specifics>
## Specific Ideas

- Packaged default artifact path should be treated as a contract target:
  `src/sol_execbench/data/amd_hardware_models/gfx1200.json`.
- v2 artifacts should not include `validation_status`; they should include
  `hardware_validation_status` and `model_validation_status`.
- Loader behavior should be strict by default and reject unknown fields.
- Compatibility APIs may remain, but hard-coded numeric hardware model fallback
  should not remain.

</specifics>

<deferred>
## Deferred Ideas

- Dataset or CLI flags for choosing an external hardware model path are Phase
  45 scope.
- Score-report warning integration beyond contract compatibility is Phase 45
  scope.
- Structured IR and operator formula work are Phases 42 and 43.
- CDNA 3 / MI300X and CDNA 4 validation remain future milestone work.

</deferred>

---

*Phase: 41-Bound Model Contract And Hardware Artifacts*
*Context gathered: 2026-05-23*
