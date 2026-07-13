# Phase 192: Official Score Evidence Contract - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 192 adds a separate official benchmark score evidence contract. It must
distinguish confirmed official score evidence from existing provisional
AMD-native derived score reports and from Trace JSONL diagnostic
`speedup_factor`.

This phase does not implement full measured baseline provenance and coverage
validation. That belongs to Phase 193. Phase 192 should define the official
score evidence surface and block placeholder/reference baselines for official
claims.

</domain>

<decisions>
## Implementation Decisions

### Evidence Surface

- **D-01:** Add a separate `official_score_evidence` schema/report instead of
  extending `amd_native_score.v1`.
- **D-02:** Existing AMD-native score reports remain derived/provisional
  evidence. They may be referenced as inputs, but they are not themselves the
  official confirmed score artifact.
- **D-03:** Diagnostic Trace JSONL `speedup_factor` must remain distinct from
  official benchmark score.

### Valid Run Preconditions

- **D-04:** A non-null official score requires measured candidate latency,
  official measured baseline latency, SOL/SOLAR bound evidence, and an explicit
  aggregation policy.
- **D-05:** If any required input is missing, the official score value must be
  null and the evidence must report stable blocker reason codes.
- **D-06:** Suite-level official evidence must report scored and unscored counts
  plus input refs so HIP can distinguish valid confirmed runs from incomplete
  evidence.

### Baseline Authority

- **D-07:** `reference_latency_ms` and placeholder/reference baseline fallback
  are blocked for confirmed/official score claims.
- **D-08:** Placeholder/reference baseline use should produce a stable blocker
  reason code rather than a provisional official score.
- **D-09:** Provisional AMD-native scores may continue to exist separately, but
  official score evidence must classify them as non-authoritative unless backed
  by official measured baseline evidence.

</decisions>

<canonical_refs>
## Canonical References

### Milestone Scope

- `.planning/REQUIREMENTS.md` — SCOR-01, SCOR-02, SCOR-03.
- `.planning/ROADMAP.md` — Phase 192 goal and success criteria.
- `.planning/research/SUMMARY.md` — score semantics and watch-outs.
- `.planning/research/ARCHITECTURE.md` — official score data flow.
- `.planning/research/PITFALLS.md` — speedup/provisional-score/baseline risks.

### Existing Code

- `src/sol_execbench/core/scoring/amd_score.py` — existing provisional
  AMD-native score schema/report and scoring helpers.
- `src/sol_execbench/core/scoring/baseline_artifact.py` — release-scoped
  scoring baseline artifact.
- `src/sol_execbench/core/scoring/amd_sol.py` and
  `src/sol_execbench/core/scoring/amd_sol_v2.py` — SOL bound evidence inputs.
- `src/sol_execbench/core/scoring/solar_derivation.py` — SOLAR aggregate
  eligibility status.
- `src/sol_execbench/core/data/trace.py` — measured latency and diagnostic
  `reference_latency_ms`/`speedup_factor` fields.

### Tests And Docs

- `tests/sol_execbench/test_amd_native_score.py` — existing provisional score
  behavior, including reference latency fallback.
- `tests/sol_execbench/test_run_dataset_amd_score.py` — dataset report wiring.
- `docs/user/EVALUATOR-CONTRACT.md`, `docs/user/RESEARCHER-GUIDE.md`,
  `docs/user/trace.md`, and `docs/user/COOKBOOK.md` — score/trace/baseline authority
  language.

</canonical_refs>

<code_context>
## Existing Code Insights

- `AmdNativeScore` already records `baseline_source`, refs, warnings, and
  `claim_level="amd-native-derived"`.
- `score_amd_native_trace_workload()` intentionally falls back to
  `trace.evaluation.performance.reference_latency_ms` when no
  `ScoringBaselineArtifact` entry exists. Phase 192 should not remove that
  existing provisional behavior.
- `ScoringBaselineArtifact` currently represents release-defined optimized
  baseline timing evidence, but it is not yet wrapped as official measured
  baseline provenance. Phase 192 can treat it as the only acceptable official
  baseline source for official score evidence, with Phase 193 adding deeper
  provenance/coverage validation.
- `sol_score(t_k, t_b, t_sol)` is the numeric score function already used by
  AMD-native score reports.

</code_context>

<specifics>
## Specific Ideas

- Add `src/sol_execbench/core/scoring/official_score.py` with dataclass models:
  workload score evidence, suite evidence, input refs, blocker reason codes,
  aggregation policy, source labels, and authority flags.
- Build official evidence from existing `AmdNativeScore` records only when
  `baseline_source == "scoring_baseline"` and score inputs are complete.
- Report blockers such as `missing_score_input`, `missing_measured_latency`,
  `missing_official_baseline`, `placeholder_baseline`, `missing_sol_bound`,
  `unscored_sol_bound`, and `missing_aggregation_policy`.
- Keep official score evidence separate from profile summary and agent feedback
  diagnostic sidecars.

</specifics>

<deferred>
## Deferred Ideas

- Full measured baseline provenance and workload coverage validation belongs to
  Phase 193.
- HIP confirmed pass/missing score/missing baseline fixture package belongs to
  Phase 194.
- Removing AMD-native score's reference latency fallback is out of scope; it
  remains useful as explicitly provisional evidence.

</deferred>

---

*Phase: 192-Official Score Evidence Contract*
*Context gathered: 2026-06-21*
