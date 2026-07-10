# Phase 193: Measured Baseline Provenance and Coverage - Context

**Gathered:** 2026-07-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 193 adds measured baseline provenance and coverage validation for confirmed
benchmark claims. It closes BASE-01 (timestamp gap on the existing measured
baseline registry), implements BASE-02 (five-state coverage validation with
stable reason codes), and wires BASE-03 (measured baseline provenance as an
authoritative baseline source for the official score gate, gated by the coverage
report).

The phase deliberately REUSES the pre-existing `baseline_export.py` measured
baseline registry shape rather than re-modelling provenance. It does NOT emit the
`official_score_evidence.v1` artifact into a run path — official score emission
stays STAGING in 193 and is wired in Phase 194 (GATE-01). HIP-facing fixtures and
consumer docs also belong to Phase 194.

</domain>

<decisions>
## Implementation Decisions

### Coverage Validation Module (BASE-02 core)

- **D-01:** Implement the coverage validator in a new module
  `src/sol_execbench/core/evidence/baseline_coverage.py` as a sibling of
  `baseline_export.py`. Do not bloat `baseline_export.py` with classification
  logic.
- **D-02:** The validator classifies each expected workload's baseline into one
  of five stable states: `confirmed`, `missing`, `stale`, `mismatched`,
  `placeholder`, each with a stable reason code.
- **D-03:** The validator accepts an explicit **current-run environment**
  parameter (hardware, ROCm version, target id, timing policy) and compares it
  against each baseline entry's recorded provenance to detect `mismatched`.
  Registry-internal-only classification (without a current-run comparison) is
  not enough to satisfy success criterion #4.
- **D-04:** `stale` is detected when the baseline entry's `trace_ref` file no
  longer exists OR the recorded provenance/generation timestamp is inconsistent
  with the current run (see D-13/D-14). A missing `trace_ref` is the primary
  stale signal.
- **D-05:** The coverage report shape is per-workload rows
  `{workload_key, state, reason_code, detail}` plus suite-level state counts,
  mirroring `OfficialScoreSuiteEvidence.blocker_summary`. State counts alone are
  insufficient.

### Stable Reason Codes (BASE-02)

- **D-06:** Emit per-state primary codes plus sub-codes so each distinct test
  case in success criterion #4 has its own code: `baseline_hardware_mismatch`,
  `baseline_timing_policy_mismatch`, `baseline_stale_trace`. States:
  `confirmed`, `missing`, `stale`, `mismatched`, `placeholder`.
- **D-07:** Naming style is snake_case with the `baseline_*` prefix, matching
  the existing official-score blockers (`missing_baseline`, etc.).
- **D-08:** Coverage `missing` and `placeholder` states map onto the EXISTING
  official-score blockers `missing_baseline` and `placeholder_baseline`
  respectively. New `stale` and `mismatched` states introduce new codes — do not
  invent a parallel namespace.
- **D-09:** `confirmed` is a positive STATUS (the passing state), not a blocker
  code. This mirrors the `status: "scored"` pattern in `OfficialScoreEvidence`.

### BASE-03 Gate Wiring (STAGING preserved)

- **D-10:** Add `measured_baseline_registry` as an accepted value in
  `DEFAULT_OFFICIAL_BASELINE_SOURCES` AND accept a coverage-validation report as
  a gate precondition parameter on
  `official_score_from_amd_native_score` /
  `build_official_score_suite_evidence`. Both changes are required.
- **D-11:** When coverage validation fails, the gate emits an umbrella blocker
  `baseline_coverage_failed` AND propagates the specific coverage reason codes
  (e.g. `baseline_hardware_mismatch`) so HIP sees precise failure reasons.
- **D-12:** Phase 193 keeps `official_score.py` STAGING — it must NOT be wired
  into a CLI command, runner, or sidecar writer in this phase. Emission into a
  run path belongs to Phase 194 (GATE-01). The aggregation-policy precondition
  remains an explicit parameter (still unresolved at the suite-report level).
- **D-13:** Keep `ScoringBaselineArtifact` (release-scoped) and the measured
  baseline registry as SEPARATE schemas. Both remain accepted
  `baseline_source` values. Do not merge them — BASE-03 requires the
  separation.

### Timestamp And Schema Versioning (BASE-01 gap)

- **D-14:** Add a top-level `generated_at` ISO-8601 UTC timestamp to the
  `baseline_export.py` registry output. Field name is `generated_at`.
- **D-15:** Reuse the existing v1.36/v1.37 freshness-identity timestamp helper
  for generating `generated_at` rather than introducing a new tz-aware helper.
- **D-16:** Keep the schema version at `baseline_registry.v1` /
  `sol_execbench.measured_baseline_registry.v1`. Adding `generated_at` is a
  backward-compatible additive field; project convention (191 decision D-07)
  keeps the schema version for additive non-breaking fields.
- **D-17:** `generated_at` feeds the `stale` detection in D-04 (timestamp age
  vs current run). It is not purely decorative.

### Claude's Discretion

- Exact dataclass names, function signatures, and the internal structure of
  `baseline_coverage.py` are at Claude's discretion as long as the decisions
  above are preserved.
- Test fixture layout and the exact set of helper functions are at Claude's
  discretion, provided all six success-criterion-#4 cases are covered.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase-Local Scope Review (authoritative re-baseline)

- `.planning/phases/193-measured-baseline-provenance-and-coverage/193-SCOPE-REVIEW.md`
  — re-baselines 193 against pre-existing `baseline_export.py`; defines the
  four-item work list and the `baseline_export.py` reuse anchor. Cite this as
  the primary scope input.

### Milestone Scope

- `.planning/REQUIREMENTS.md` — BASE-01, BASE-02, BASE-03 and traceability.
- `.planning/ROADMAP.md` — Phase 193 goal and success criteria.
- `.planning/research/SUMMARY.md`, `.planning/research/ARCHITECTURE.md`,
  `.planning/research/PITFALLS.md` — confirmed-evidence data flow and risks.

### Prior Phase Decisions (locked, do not re-litigate)

- `.planning/phases/192-official-score-evidence-contract/192-CONTEXT.md` —
  decisions D-07/D-08 block `reference_latency_ms` and placeholder/reference
  baseline fallback for official claims; defines official-score blockers.

### Existing Code

- `src/sol_execbench/core/evidence/baseline_export.py` — REUSE ANCHOR.
  `export_hip_baseline_registry()` emits `baseline_registry.v1` /
  `sol_execbench.measured_baseline_registry.v1` with provenance and workload
  coverage set. Extend this shape; do not duplicate it.
- `src/sol_execbench/core/scoring/official_score.py` — STAGING official score
  gate. `official_score_from_amd_native_score()`,
  `build_official_score_suite_evidence()`, existing blockers, and
  `DEFAULT_OFFICIAL_BASELINE_SOURCES`.
- `src/sol_execbench/core/scoring/baseline_artifact.py` — release-scoped
  `ScoringBaselineArtifact` (kept separate per D-13).
- `src/sol_execbench/core/evidence/baseline.py` — `compare_trace_baselines`
  win/parity/loss classification (different concern; do not conflate).

### Tests And Docs

- `tests/sol_execbench/core/evidence/test_baseline_export.py` — existing
  registry/CLI/coverage tests (extend, do not break).
- `tests/sol_execbench/core/evidence/test_baseline_comparison.py` — existing
  comparison tests (orthogonal; success-criterion-#4 coverage cases are new).
- `docs/EVALUATOR-CONTRACT.md`, `docs/trace.md` — baseline/score authority
  language (HIP-facing fixture/doc updates belong to 194).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `export_hip_baseline_registry()` already records per-entry provenance
  (hardware, rocm_version, sol_version, target_id, timing_policy), workload
  coverage set (`expected_workload_keys`), and a binary `coverage_status`. This
  is the provenance shape to reuse for BASE-02 classification.
- `OfficialScoreEvidence` / `OfficialScoreSuiteEvidence` already model
  per-workload + suite-level blocker summaries and `score_authority` — the
  coverage report should mirror this layering.
- Existing official-score blockers (`missing_baseline`, `placeholder_baseline`)
  are the stable codes to map onto for D-08.

### Established Patterns

- Strict authority boundaries: canonical Trace JSONL remains the authority;
  evidence contracts are machine-verifiable but stay separate from diagnostic
  sidecars.
- Stable snake_case reason codes for cross-report consistency.
- Backward-compatible additive schema fields keep the schema version (191 D-07).
- STAGING modules declare their unwired status in the module docstring
  (`official_score.py` is the template).

### Integration Points

- `baseline_export.py` registry output gains `generated_at` (D-14).
- New `baseline_coverage.py` consumes the registry shape + a current-run
  environment to produce the coverage report.
- `official_score.py` gate accepts the coverage report as a precondition and
  adds `measured_baseline_registry` as an accepted baseline source (D-10/D-11),
  while remaining STAGING (D-12).

</code_context>

<specifics>
## Specific Ideas

- Treat the 193-SCOPE-REVIEW.md four-item work list as the plan task breakdown:
  1) `generated_at` timestamp (BASE-01); 2) five-state coverage report + reason
  codes (BASE-02); 3) six coverage test cases — complete, partial, hardware
  mismatch, timing-policy mismatch, stale trace pointer, placeholder rejection
  (BASE-02 criterion #4); 4) gate wiring with schemas kept separate (BASE-03).
- Cite `baseline_export.py` explicitly as the reuse anchor in the plan.
- Keep all changes CPU-safe and unit-testable without a real GPU.

</specifics>

<deferred>
## Deferred Ideas

- Wiring `official_score_evidence.v1` emission into a CLI/runner/sidecar run
  path — Phase 194 (GATE-01).
- HIP-facing confirmed/missing/placeholder/profiler-partial/diagnostic-only
  fixtures and consumer docs — Phase 194 (GATE-02).
- Resolving the suite-report aggregation-policy concept — remains an explicit
  parameter in 193; full resolution may land with 194 emission wiring.
- Removing AMD-native score's `reference_latency_ms` fallback — out of scope; it
  remains explicitly provisional evidence.

</deferred>

---

*Phase: 193-Measured Baseline Provenance and Coverage*
*Context gathered: 2026-07-10*
