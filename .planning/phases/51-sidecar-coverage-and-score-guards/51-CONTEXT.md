# Phase 51: Sidecar Coverage And Score Guards - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase turns the Phase 48-50 derivation evidence into inspectable internal
SOLAR sidecar coverage and score-guard semantics. Users should be able to parse
sidecars and AMD-native score reports to distinguish `scored`, `degraded`, and
`unscored` derivation evidence without manually interpreting warning text.

This phase is not the public runner/documentation closure phase. It should not
change canonical benchmark schemas, primary CLI behavior, canonical trace JSONL,
or public documentation claims. Public docs, dataset-runner reporting closure,
and final claim-boundary docs remain Phase 52.

</domain>

<decisions>
## Implementation Decisions

### Sidecar Coverage Scope
- Aggregate existing SOLAR derivation semantic groups into family-aware coverage
  evidence.
- Coverage must expose extraction provenance, missing patterns, unsupported
  patterns, degraded nodes, estimated nodes, family counts, and status counts.
- Coverage should reuse Phase 48-50 group-local formula, byte, bound,
  confidence, warning, and missing-evidence records.
- Do not re-run extraction from candidate solution code and do not create a
  second derivation graph.

### Machine-Verifiable Status Semantics
- Aggregate SOLAR evidence must expose explicit `scored`, `degraded`, and
  `unscored` states.
- `scored` requires supported groups with complete score-eligible evidence.
- `degraded` preserves warning and missing-evidence detail when evidence is
  useful but incomplete.
- `unscored` carries explicit unsupported/unscored evidence and must be
  distinguishable from missing sidecar data.

### AMD-Native Score Guard Semantics
- AMD-native scoring should return `None` for unscored SOLAR evidence.
- AMD-native score reports should preserve warnings for degraded SOLAR evidence.
- Degraded evidence must not be silently treated as fully scored.
- This phase may wire internal sidecar coverage into score guard behavior, but
  must preserve AMD-native-derived claim boundaries and avoid public schema
  drift.

### Parse And Serialize Coverage
- TEST-03 requires sidecar parse/serialize round-trip tests for every new
  machine-verifiable derivation evidence field.
- New coverage/aggregate/score-guard sidecar fields must use strict exact-key
  parser behavior, deterministic ordering, and JSON-safe values.
- Existing Phase 48-50 evidence parser coverage must remain green.

### Public Boundary
- Keep canonical `Definition`, `Workload`, `Trace`, primary CLI help, and
  canonical trace JSONL unchanged.
- Do not add public primary CLI options for SOLAR derivation coverage in this
  phase.
- Do not add new dependencies, paper-scale dataset claims, hardware-validation
  claims, hosted leaderboard claims, or NVIDIA equivalence claims.

### the agent's Discretion
- Exact internal dataclass names, helper names, and sidecar field names are at
  the agent's discretion if they remain deterministic, parseable, and consistent
  with existing scoring conventions.
- The planner may split work by coverage model, score guard integration, and
  parser/guardrail closure.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/scoring/solar_derivation.py` contains internal SOLAR
  derivation sidecar evidence, strict parser helpers, confidence/status
  mapping, group-local formula/byte/bound evidence, and family-specific
  missing/warning logic.
- `src/sol_execbench/core/scoring/amd_sol.py`,
  `src/sol_execbench/core/scoring/amd_sol_v2.py`, and
  `src/sol_execbench/core/scoring/amd_score.py` contain existing AMD-native SOL
  artifacts, bound reports, and score eligibility behavior.
- `tests/sol_execbench/test_solar_derivation_evidence.py` covers internal
  sidecar parsing and round-trips.
- `tests/sol_execbench/test_public_contract_guardrails.py` guards canonical
  schemas, CLI, trace JSONL, and AMD-native score boundaries.
- `tests/sol_execbench/test_amd_sol_v2.py` covers AMD SOL v2 artifact behavior.

### Established Patterns
- Internal artifacts use frozen dataclasses, explicit `to_dict()` methods, and
  strict exact-key parser helpers.
- Status values should remain JSON-safe strings and deterministic.
- Public benchmark contracts stay unchanged unless the roadmap explicitly says
  otherwise.
- Tests should include positive, degraded, unscored, parser rejection, and
  public boundary guardrails.

### Integration Points
- Coverage should be derived from `SolarDerivationEvidence.groups`, not from a
  parallel extractor.
- Score guard integration should consume sidecar statuses and warnings while
  preserving existing AMD-native score artifacts and claim boundaries.
- Phase 49/50 family modeling tests must remain green.

</code_context>

<specifics>
## Specific Ideas

- Treat coverage as a compact aggregate over existing groups: families,
  statuses, missing evidence, warnings, estimated nodes, degraded nodes, and
  unsupported nodes.
- Preserve traceability from coverage records back to group IDs and node IDs.
- Keep `None` score behavior specific to unscored SOLAR evidence; degraded
  evidence should be represented as degraded with warnings rather than hidden.
- Prefer narrow internal tests over public UI changes.

</specifics>

<deferred>
## Deferred Ideas

- Dataset-runner reporting closure and user-facing documentation remain Phase
  52.
- Public claim guardrails for paper parity, hardware validation, hosted
  leaderboard readiness, and NVIDIA equivalence are finalized in Phase 52.
- Real hardware validation and paper-scale extraction remain out of scope.
</deferred>
