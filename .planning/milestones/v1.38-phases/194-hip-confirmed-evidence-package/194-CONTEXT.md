# Phase 194: HIP Confirmed Evidence Package - Context

**Gathered:** 2026-07-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 194 publishes the HIP-facing confirmed-evidence package: the evaluator
contract advertises confirmed benchmark evidence capabilities (GATE-01), HIP-facing
fixtures cover the six confirmed/missing/placeholder/profiler-partial/diagnostic-only
cases (GATE-02), and a SOL-provided emission path lets HIP remove
`missing_score`/`missing_baseline`/`placeholder_baseline` blockers for valid runs
while keeping precise blocker codes for invalid runs (GATE-03).

This phase wires the official score gate (STAGING in 193) into a SOL-provided
emission surface and publishes the contract + fixtures + docs HIP needs for
cutover-gate decisions. It does NOT change canonical Trace JSONL, does NOT
promote diagnostic sidecars to confirmed authority, and does NOT perform real
GPU validation.

</domain>

<decisions>
## Implementation Decisions

### GATE-01 — Evaluator Contract Capability Surface

- **D-01:** Add confirmed capabilities to `build_evaluator_contract()`:
  `official_score.evidence: "confirmed"` and
  `measured_baseline.coverage: "confirmed"`. Keep `profiling.evidence: "optional"`
  and the existing diagnostic sidecar capabilities unchanged.
- **D-02:** Add confirmed boundaries to the contract `boundaries` list:
  `{"owner": "sol", "scope": "official_score", "authority": "confirmed"}` and
  `{"owner": "sol", "scope": "measured_baseline", "authority": "confirmed"}`.
  Keep all diagnostic sidecar boundaries (`agent_feedback`, `profile_summary`,
  `environment_budget`, `static_resource_footprint`, `decision`) at
  `authority: "diagnostic"`.
- **D-03:** The contract advertises the stable blocker reason-code vocabulary
  (e.g. `missing_score`, `missing_measured_latency`, `missing_baseline`,
  `placeholder_baseline`, `missing_sol_bound`, `missing_aggregation_policy`,
  `baseline_coverage_failed`) so HIP knows exactly which blockers SOL can emit
  and remove.
- **D-04:** Keep the contract schema version at `sol_execbench.evaluator_contract.v2`
  — new capabilities/boundaries/vocabulary are additive (project convention).

### GATE-02 — HIP-Facing Fixtures

- **D-05:** New fixture directory
  `tests/sol_execbench/fixtures/confirmed_evidence/`.
- **D-06:** One bundle JSON per case combining the official_score_evidence
  payload + the measured baseline registry summary + the coverage summary, so
  HIP consumes one coherent bundle per scenario.
- **D-07:** Six required cases, each a `<case>.bundle.json` plus a
  `<case>.case.json` metadata file (mirroring the existing sidecar fixture
  convention): `confirmed-pass`, `missing-score`, `missing-baseline`,
  `placeholder-baseline`, `profiler-partial`, `diagnostic-only-sidecar`.
- **D-08:** A loader test loads each bundle and asserts its blocker reason-code
  set matches the GATE-03 expected set (empty for confirmed-pass; precise codes
  for the failure cases).

### GATE-03 — Emission And Blocker Removal

- **D-09:** Add a SOL CLI command `sol-execbench official-score` that emits an
  `official_score_evidence.v1` JSON artifact from inputs, mirroring the
  agent_feedback/profile_summary sidecar-producer pattern. It loads a trace /
  AMD-native score input, a scoring baseline, and a measured baseline registry,
  builds the coverage report, runs the gate, and writes the evidence JSON.
- **D-10:** `--aggregation-policy` is a REQUIRED CLI flag (the caller supplies
  the policy). This resolves the 193 STAGING note's unresolved aggregation-policy
  decision without adding the concept to `AmdNativeSuiteReport`.
- **D-11:** The CLI builds the coverage report from the measured baseline
  registry plus a `--current-run-env` (hardware / rocm / target / timing-policy)
  input, then passes it to `build_official_score_suite_evidence()` as the
  `coverage_report` precondition.
- **D-12:** A GATE-03 round-trip test uses the fixtures (or CLI invocations) to
  prove: a valid confirmed run emits evidence with NO `missing_score` /
  `missing_baseline` / `placeholder_baseline` blockers; invalid runs keep their
  precise blocker reason codes.

### Documentation

- **D-13:** Extend `docs/user/EVALUATOR-CONTRACT.md` — expand `## Official Score
  Evidence`, `## Capabilities`, and `## Ownership Boundary` with the confirmed
  evidence surface, the blocker reason-code vocabulary, and the confirmed
  boundaries.
- **D-14:** Add `docs/user/confirmed_evidence.md` — a HIP-facing consume/diagnostic-only
  table explaining which SOL artifacts HIP must consume for confirmed pass/fail
  (official_score_evidence + measured baseline coverage) and which remain
  diagnostic-only (agent_feedback, profile_summary, decision sidecars).
- **D-15:** Document the six fixture cases in `docs/user/confirmed_evidence.md`
  (intent + expected blocker set per case).
- **D-16:** Reaffirm authority wording: diagnostic sidecars cannot provide
  confirmed pass/fail authority; only official_score + measured baseline
  coverage can.

### Claude's Discretion

- Exact CLI option names, the bundle JSON schema details, and fixture content
  are at Claude's discretion as long as the decisions above are preserved.
- The CLI may reuse existing loaders (`load_scoring_baseline_artifact`,
  `export_hip_baseline_registry` output, `build_amd_native_suite_report`) where
  practical rather than inventing new input parsers.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone Scope

- `.planning/REQUIREMENTS.md` — GATE-01, GATE-02, GATE-03 and traceability.
- `.planning/ROADMAP.md` — Phase 194 goal and success criteria.

### Prior Phase Decisions (locked, do not re-litigate)

- `.planning/phases/193-measured-baseline-provenance-and-coverage/193-CONTEXT.md`
  — measured baseline registry shape, five-state coverage validator, and the
  official score gate's `coverage_report` precondition (BASE-03).
- `.planning/phases/193-measured-baseline-provenance-and-coverage/193-VERIFICATION.md`
  — what 193 shipped (gate stays STAGING until 194 wires emission).
- `.planning/phases/192-official-score-evidence-contract/192-CONTEXT.md` —
  official score blocker vocabulary and reference-latency blocking (D-07/D-08).

### Existing Code

- `src/sol_execbench/core/data/contract.py` — `build_evaluator_contract()` /
  `EvaluatorContract` (GATE-01 anchor; capabilities, boundaries,
  baseline_export_fields).
- `src/sol_execbench/core/scoring/official_score.py` — STAGING gate to wire
  emission around; `build_official_score_suite_evidence()`,
  `official_score_from_amd_native_score()`, blocker constants,
  `BASELINE_COVERAGE_FAILED_BLOCKER`.
- `src/sol_execbench/core/evidence/baseline_coverage.py` —
  `validate_baseline_coverage()`, `CurrentRunEnvironment`, `BaselineCoverageReport`.
- `src/sol_execbench/core/evidence/baseline_export.py` — measured baseline
  registry producer (registry shape + `generated_at`).
- `src/sol_execbench/core/scoring/amd_score.py` — `AmdNativeScore` /
  `build_amd_native_suite_report*` (input to the official score gate).
- `src/sol_execbench/cli/commands/metadata.py` — `contract` / `doctor` /
  `toolchain` GPU-free CLI commands (pattern reference for the new
  `official-score` command).
- `src/sol_execbench/cli/commands/baseline.py` — existing `baseline export` CLI
  (pattern reference for loading baseline inputs).

### Tests And Docs

- `tests/sol_execbench/fixtures/agent_feedback/` and
  `tests/sol_execbench/fixtures/profile_summary/` — HIP-facing fixture naming
  convention (`<state>.<artifact>.json` + `.case.json`).
- `tests/sol_execbench/core/evidence/test_official_score_evidence.py` — gate
  behavior tests to extend for the GATE-03 round-trip.
- `docs/user/EVALUATOR-CONTRACT.md` — contract doc to extend (already has
  `## Official Score Evidence` section).
- `docs/user/profile_summary_sidecar.md`, `docs/user/agent_feedback_sidecar.md` —
  diagnostic-sidecar doc conventions to mirror for `docs/user/confirmed_evidence.md`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `build_evaluator_contract()` already centralizes capabilities, boundaries, and
  field groups — GATE-01 is an additive extension.
- `build_official_score_suite_evidence()` already accepts `coverage_report` and
  emits the evidence dict (193); the CLI is a thin loader + writer around it.
- `validate_baseline_coverage()` + `CurrentRunEnvironment` already produce the
  coverage report from a registry (193); the CLI builds the env from flags.
- The agent_feedback/profile_summary sidecar producers are the established
  pattern for a SOL-provided optional artifact emitter.

### Established Patterns

- GPU-free CLI commands live in `src/sol_execbench/cli/commands/` and emit
  `--json` output (metadata.py pattern).
- HIP-facing fixtures use `<state>.<artifact>.json` + `<state>.<artifact>.case.json`
  in `tests/sol_execbench/fixtures/<artifact>/`.
- Diagnostic sidecars are explicitly marked `authority: "diagnostic"` in the
  contract boundaries; confirmed evidence gets `authority: "confirmed"`.
- Contract schema stays at v2 for additive capability changes.

### Integration Points

- `build_evaluator_contract()` gains confirmed capabilities + boundaries + the
  blocker vocabulary.
- New `official-score` CLI command in `cli/commands/` registered via the CLI
  root.
- New `confirmed_evidence/` fixture directory + a loader test.
- `docs/user/EVALUATOR-CONTRACT.md` extended; new `docs/user/confirmed_evidence.md`.

</code_context>

<specifics>
## Specific Ideas

- Treat the four GATE requirement clusters as the plan task breakdown:
  1) GATE-01 contract surface (D-01..D-04); 2) GATE-03 CLI emission + round-trip
  test (D-09..D-12); 3) GATE-02 fixtures + loader test (D-05..D-08); 4) docs
  (D-13..D-16). Ordering: contract first (capabilities advertise what the CLI
  emits), then CLI, then fixtures (which the CLI/round-trip validates), then docs.
- The CLI's `--aggregation-policy` required flag is the explicit resolution of
  the 193 STAGING note — cite that in the plan.
- Keep everything CPU-safe and fixture-backed; no real GPU required.

</specifics>

<deferred>
## Deferred Ideas

- Wiring `official_score_evidence.v1` into the default eval-driver / dataset
  runner run path (so every run emits it automatically) — this phase adds the
  explicit CLI emission surface; default-run auto-emission is a follow-up if
  adoption demands it.
- Deriving the aggregation policy from `AmdNativeSuiteReport` (currently an
  explicit CLI flag).
- Real GPU validation of the confirmed-evidence path.
- Promoting any diagnostic sidecar to confirmed authority.

</deferred>

---

*Phase: 194-HIP Confirmed Evidence Package*
*Context gathered: 2026-07-10*
