# Phase 52: Dataset Runner And Public Contract Closure - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning
**Mode:** `$gsd-autonomous --auto` recommended defaults

<domain>
## Phase Boundary

This phase closes v1.10 by connecting the completed SOLAR derivation sidecars
and AMD-native score guards to the intended runner, report, documentation, and
guardrail surfaces. The result should be usable and auditable without changing
canonical benchmark contracts.

This phase is not a paper-scale extraction phase and not a hardware-validation
phase. It must not claim original-paper 124-model / 235-problem coverage,
NVIDIA Blackwell or B200 equivalence, hosted leaderboard readiness, CDNA 3 /
MI300X / CDNA 4 validation, NVFP4/MXFP4 validation, or new real-hardware
results.

</domain>

<decisions>
## Implementation Decisions

### Runner Surface
- Use the existing dataset-runner derived artifact pattern instead of adding
  primary `sol-execbench` CLI behavior.
- If runner integration is needed, keep it opt-in and sidecar/report oriented:
  derived SOLAR sidecars and AMD-native score reports may be written as
  separate artifacts, while canonical trace JSONL remains unchanged.
- Do not execute submitted candidate solution code to derive SOLAR evidence.
  Derivation must continue to use canonical problem/reference/workload inputs
  and existing static evidence paths.
- Dataset-runner behavior should remain compatible when no SOLAR derivation
  sidecar is requested or available.

### Derived Report Evidence References
- Derived reports must keep `claim_level` at `amd-native-derived`.
- Evidence references should make formula evidence, hardware model evidence,
  coverage evidence, and score eligibility auditable in derived artifacts.
- Public score `evidence_refs` must preserve existing public keys unless a
  derived report-specific internal field is explicitly scoped and guarded.
- The runner and report should distinguish missing sidecar data from explicit
  `unscored` SOLAR evidence.

### Documentation Surface
- Documentation should explain how to consume v1.10 SOLAR sidecars and
  AMD-native score guard outputs from local derived reports.
- Documentation must state that v1.10 sidecars are AMD ROCm derived evidence,
  not paper-scale dataset extraction, not upstream SOLAR parity, not NVIDIA
  B200/Blackwell equivalence, not hosted leaderboard readiness, and not new
  real-hardware validation.
- Prefer updating existing analysis/internal docs over adding a broad new
  public tutorial unless the planner finds a small dedicated doc clearer.

### Public Contract Guardrails
- Preserve canonical `Definition`, `Workload`, `Trace`, primary CLI help,
  canonical trace JSONL, existing public benchmark semantics, and dependency
  set.
- TEST-04 should prove public contracts remain unchanged even after runner and
  derived report closure.
- Guardrails should inspect exact JSON keys where possible to avoid false
  positives from established v2 fields such as `coverage_summary`.

### Claim Guardrails
- TEST-05 should prevent v1.10 artifacts and docs from implying:
  paper benchmark parity, paper-scale 124-model / 235-problem extraction,
  NVIDIA Blackwell or B200 equivalence, hosted leaderboard readiness, CDNA 3 /
  MI300X / CDNA 4 validation, NVFP4/MXFP4 validation, or new real-hardware
  validation.
- Claim guardrails should preserve positive AMD-local language: v1.10 provides
  paper-aligned automatic SOLAR derivation evidence for the ROCm port.
- Avoid banning historical mentions of B200 or original-paper context when
  those mentions are explicitly framed as not claimed or out of scope.

### the agent's Discretion
- Exact option names, helper names, report field names, and doc placement are
  at the agent's discretion if they remain deterministic, local, and guarded.
- The planner may split work into runner/report integration, documentation,
  and public/claim guardrail closure.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/run_dataset.py` already writes derived AMD-native score reports and
  v2 AMD SOL bound sidecars without mutating canonical trace JSONL.
- `src/sol_execbench/core/scoring/amd_score.py` now accepts optional internal
  SOLAR derivation/aggregate score guards and preserves public claim level.
- `src/sol_execbench/core/scoring/solar_derivation.py` now emits strict
  `coverage_summary` and `aggregate_status` sidecars.
- `docs/analysis.md` already documents derived AMD-native dataset score
  reports, sidecar directories, and claim boundaries.
- `docs/internal/solar_derivation_contract.md` documents the v1.10 derivation
  contract and forbidden claim phrases.
- `tests/sol_execbench/test_run_dataset_amd_score.py` covers dataset-runner
  derived AMD score reports and sidecar behavior.
- `tests/sol_execbench/test_public_contract_guardrails.py` guards public
  schemas, CLI, trace JSONL, and score evidence refs.
- `tests/sol_execbench/test_v1_9_validation_closure.py` and
  `tests/sol_execbench/test_solar_derivation_contract.py` show existing claim
  guardrail patterns.

### Established Patterns
- Derived artifacts are opt-in report/sidecar files separate from canonical
  trace JSONL.
- Public contracts are protected with static tests that inspect schema dumps,
  CLI help text, trace serialization, docs, and exact JSON keys.
- Claim guardrails should be explicit and evidence-based rather than relying on
  generic substring bans that catch legitimate historical context.
- Tests should remain deterministic and should not require ROCm hardware or
  candidate execution.

### Integration Points
- Dataset score report generation currently flows through
  `score_amd_native_trace_workload()` and `build_amd_native_suite_report()`.
- SOLAR derivation sidecars can be consumed as internal `SolarDerivationEvidence`
  or aggregate status values for score guards.
- Report evidence refs need to tie back to formula/bound evidence, hardware
  model refs, coverage sidecars, and score eligibility without widening
  canonical schemas.

</code_context>

<specifics>
## Specific Ideas

- Add an opt-in runner path for reading or writing SOLAR derivation sidecars
  alongside AMD-native score reports, with workload UUID keyed propagation into
  score guards.
- Add a derived report metadata block or sidecar manifest if needed to expose
  formula, hardware model, coverage, and score-eligibility evidence refs while
  preserving canonical trace JSONL and primary CLI.
- Update docs to show a minimal local workflow: derive/run dataset, inspect
  sidecar `coverage_summary` and `aggregate_status`, consume AMD-native score
  warnings, and understand claim limits.
- Extend public and claim guardrail tests to cover v1.10 docs, runner output
  shape, derived report refs, and forbidden overclaims.

</specifics>

<deferred>
## Deferred Ideas

- Original-paper 124-model / 235-problem extraction and curation remains
  future work.
- MI300X, CDNA 3, CDNA 4, NVFP4, and MXFP4 real-hardware validation remain
  future work.
- Hosted leaderboard or submission service remains future work.
- NVIDIA Blackwell/B200 comparison methodology, if ever desired, must be a
  separate non-ROCm claim analysis effort.
</deferred>
