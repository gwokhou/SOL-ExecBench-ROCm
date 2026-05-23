# Phase 47: Derivation Contract And Golden Fixture Matrix - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase establishes the v1.10 SOLAR derivation contract and fixture matrix before implementation expands extraction or modeling behavior. It must define expected family recognition, degradation, negative behavior, claim boundaries, and machine-readable fixture expectations for later phases.

This phase does not implement real extractor/modeling logic. Production scoring code should remain unchanged unless a pure constant or type export is clearly required.

</domain>

<decisions>
## Implementation Decisions

### Fixture Matrix Scope
- Store fixtures under `tests/sol_execbench/fixtures/solar_derivation/` as JSON fixtures with a Python helper loader.
- Cover each target family with at least three fixture classes: positive, degraded, and unsupported or negative. Families are attention, MoE, convolution, SSM/Mamba, embedding or positional, and linear projection.
- Fixture expectations should record expected family, subroles, SOLAR state, required evidence, missing evidence, and stable warning prefixes. Phase 47 should not require complete FLOP or byte golden numbers.
- The fixture matrix must explicitly verify paper-aligned derivation behavior without claiming paper-scale dataset extraction, hosted leaderboard readiness, NVIDIA Blackwell/B200 equivalence, or new real-hardware validation.

### Contract Artifact Shape
- Add `docs/internal/solar_derivation_contract.md` covering the sidecar-only contract, family states, fixture schema, and claim boundaries.
- Add a test-side helper that loads fixture JSON and validates required base fields so the contract is executable.
- Reuse existing confidence and aggregate vocabulary: `SUPPORTED`, `INEXACT`, `UNSUPPORTED`, and `scored`, `degraded`, `unscored`.
- Express negative fixtures with `expected_status`, `missing_evidence`, and stable warning prefixes rather than exception expectations.

### Phase 47 Execution Boundary
- Do not implement real extractor or modeling changes in Phase 47. Leave extraction and modeling implementation for Phases 48-50.
- Tests should validate fixture schema completeness, coverage across all required families and states, claim-boundary documentation, and fixture expectations that later phases can consume.
- Avoid production scoring-code changes unless a pure constant or type export is needed. Prefer docs/internal, fixtures, and tests.
- Phase 47 is complete when phase artifacts, the internal contract doc, fixtures, loader tests, and focused pytest checks are committed.

### the agent's Discretion
- The exact JSON field names and loader helper names are at the agent's discretion as long as they are stable, clear, and aligned with existing pytest conventions.
- The exact split between one fixture file per case and grouped fixture files is at the agent's discretion, provided tests can identify family/state coverage precisely.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Scoring code lives under `src/sol_execbench/core/scoring/`, especially `amd_bound_graph.py`, `amd_bound_estimates.py`, `amd_sol.py`, `amd_sol_v2.py`, and `amd_score.py`.
- Existing scoring tests include `tests/sol_execbench/test_amd_bound_graph.py`, `tests/sol_execbench/test_amd_bound_estimates.py`, `tests/sol_execbench/test_amd_sol_v2.py`, `tests/sol_execbench/test_amd_native_score.py`, and `tests/sol_execbench/test_public_contract_guardrails.py`.
- v1.9 validation closure tests already check bound-modeling coverage and public contract guardrails.

### Established Patterns
- Tests live under `tests/sol_execbench/` and use pytest with descriptive `test_*` names, parametrization, native assertions, and local helper functions.
- File-based test data should be loaded from repository test fixtures or `tmp_path`, with real JSON serialization/parsing tested instead of mocked.
- Public contract and claim boundaries are enforced by explicit guardrail tests rather than relying on docs alone.
- Python files should use repository Ruff style, SPDX/license headers where consistent with nearby tests, and clear module docstrings for non-trivial test modules.

### Integration Points
- New fixture JSON should live under `tests/sol_execbench/fixtures/solar_derivation/`.
- New contract documentation should live under `docs/internal/solar_derivation_contract.md`.
- New executable contract coverage should live under `tests/sol_execbench/test_solar_derivation_contract.py` or a similarly focused test module.
- Public contract guardrails may need targeted assertions that the new SOLAR contract does not alter canonical schema or CLI claims.

</code_context>

<specifics>
## Specific Ideas

- Keep Phase 47 focused on contract and fixtures only; implementation phases should consume these artifacts later.
- Use machine-readable fields for expected family, subroles, status, missing evidence, and warning prefixes.
- Explicitly preserve AMD-native-derived claim boundaries and paper-aligned-but-not-paper-equivalent wording.

</specifics>

<deferred>
## Deferred Ideas

- Real extraction infrastructure is deferred to Phase 48.
- High-confidence family formula implementation is deferred to Phase 49.
- MoE and SSM/Mamba modeling implementation is deferred to Phase 50.
- Sidecar coverage and score guard integration is deferred to Phase 51.
</deferred>
