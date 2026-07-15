# Confirmed Evidence Consumer Guide (HIP)

This document explains which SOL artifacts HIP must consume to make **confirmed
benchmark pass/fail** decisions and which artifacts remain **diagnostic-only**.
It is the HIP-facing companion to `docs/user/EVALUATOR-CONTRACT.md`.

## Authority Model

Only two SOL surfaces carry **confirmed** authority for cutover-gate decisions:

| Surface | Capability key | Authority | What it proves |
| --- | --- | --- | --- |
| Official score evidence (`official_score_evidence.v1`) | `official_score.evidence` | `confirmed` | Confirmed benchmark score (non-null only when all gate inputs are present). |
| Measured baseline coverage (`BaselineCoverageReport`) | `measured_baseline.coverage` | `confirmed` | Confirmed measured baseline provenance + five-state coverage validation. |

Everything else SOL emits is **diagnostic-only** and CANNOT satisfy a confirmed
pass/fail decision:

| Artifact | Capability key | Authority |
| --- | --- | --- |
| Agent feedback sidecar (`sol_execbench.agent_feedback.v2`) | `agent_feedback.sidecar` | `diagnostic` |
| Profile summary sidecar (`sol_execbench.profile_summary.v2`) | `profile_summary.sidecar` | `diagnostic` |
| Environment budget sidecar | `environment_budget.sidecar` | `diagnostic` |
| Static resource footprint sidecar | `static_resource_footprint.sidecar` | `diagnostic` |
| Decision sidecar (`sol_execbench.decision.v1`) | `decision.sidecar` | `diagnostic` |
| Trace JSONL `speedup_factor` | (trace field) | diagnostic ratio only |

Diagnostic sidecars may guide a next experiment, but they cannot promote
correctness, timing, score, evidence-tier, release-gate, cutover, paper-parity,
or leaderboard authority.

## Consuming Confirmed Evidence

For the producer-side artifact chain—hardware calibration and fusion evidence,
AMD-native score generation, frozen release baseline, independent rerun, and
this final command—see [Release and Official Score Workflow](RELEASE-SCORING.md).
This guide defines what a consumer may accept after the command runs; it does
not make a diagnostic sidecar or an incomplete artifact set confirmed evidence.

### Emission (GATE-03)

SOL emits `official_score_evidence.v1` via the GPU-free CLI:

```bash
sol-execbench score official \
  --amd-native-score amd-native-score.json \
  --scoring-baseline scoring-baseline.json \
  --release-baseline-bundle release-baseline-bundle.json \
  --release-baseline-verification release-baseline-verification.json \
  --suite-manifest suite.json \
  --aggregation-policy "fixed_suite_denominator_zero_for_blocked" \
  --current-run-env-hardware gfx1200 \
  --current-run-env-rocm 7.1 \
  --current-run-env-target attention \
  --current-run-env-timing-policy latency_ms \
  --output official-score-evidence.json
```

`--aggregation-policy` is **required** and must be
`fixed_suite_denominator_zero_for_blocked`; the caller supplies this explicit
policy without adding the concept to `AmdNativeSuiteReport`. The
`--current-run-env-*` flags build the coverage report's current-run environment;
omit any flag to skip that field's comparison.

### Blocker Removal (GATE-03)

A valid SOL run removes the three HIP cutover blockers:

- `missing_score`
- `missing_baseline`
- `placeholder_baseline`

...when the emitted `official_score_evidence.v1` has `score_authority: true`
(i.e. `scored_count > 0` and `unscored_count == 0` and no `blocker_summary`
entries). Invalid runs keep their precise `blocker_reason_codes` so HIP can
report exactly what is missing. The full stable vocabulary is advertised on the
contract as `confirmed_evidence_blockers`:

- `missing_score`
- `missing_measured_latency`
- `missing_baseline`
- `placeholder_baseline`
- `missing_sol_bound`
- `missing_aggregation_policy`
- `baseline_coverage_failed` (emitted with propagated coverage sub-codes such as
  `baseline_hardware_mismatch`, `baseline_timing_policy_mismatch`, and
  `baseline_stale_trace` when measured baseline coverage does not fully confirm)

### Release Baseline Is Required

`reference_latency_ms` and placeholder/reference baseline fallback are blocked
for confirmed official score claims. A baseline sourced from
`reference_latency` produces a `placeholder_baseline` blocker, not a confirmed
score. Confirmed baseline authority requires a `scoring_baseline` entry whose
digest matches `release_baseline_bundle.v1` and whose workload is `official` in
both the bundle and its independent `release_baseline_verification.v1`.
The cited bound and hardware-model files must match the bundle's per-workload
SHA-256 values, the bound schema must be `sol_execbench.amd_sol_bound.v5`, and
the embedded model must use `sol_execbench.amd_hardware_model.v3`. The bound
also cites checksum-verified fusion-validation evidence.
Every release bundle also declares a non-empty `scope`; consumers must preserve
that scope rather than describing an authority slice as a full suite.
`measured_baseline_registry` remains useful for coverage diagnostics, but it
cannot supply `T_b`.

## Fixture Cases

HIP can validate its consumption against six machine-verifiable fixture bundles
in `tests/sol_execbench/fixtures/confirmed_evidence/`. Each `<case>.bundle.json`
combines an `official_score_evidence` payload, a measured baseline registry
summary, a coverage summary, and (for diagnostic cases) referenced diagnostic
sidecars. Each `<case>.case.json` records the expected blocker set and score
authority.

| Case | Expected blockers | Score authority | Notes |
| --- | --- | --- | --- |
| `confirmed-pass` | _(none)_ | true | Valid SOL run; HIP confirms pass. |
| `missing-score` | `missing_score` | false | Missing score input. |
| `missing-baseline` | `missing_baseline` | false | Missing official baseline. |
| `placeholder-baseline` | `placeholder_baseline` | false | Reference-latency-derived baseline is not measured. |
| `profiler-partial` | `missing_score` | false | Official score blocked AND a partial `profile_summary` sidecar present; the diagnostic sidecar does NOT remove the blocker. |
| `diagnostic-only-sidecar` | `missing_baseline`, `missing_score` | false | Only diagnostic sidecars present, no confirmed evidence; blockers remain. |

The loader test
`tests/sol_execbench/core/evidence/test_confirmed_evidence_fixtures.py` asserts
each bundle's blocker set matches its `expected_blockers` and that diagnostic
sidecar presence never removes a confirmed-evidence blocker.

## Authority Wording (Reaffirmed)

Diagnostic sidecars (`agent_feedback`, `profile_summary`, `environment_budget`,
`static_resource_footprint`, `decision`) are optional adapter inputs only. They
cannot provide confirmed pass/fail, release-gate, cutover, paper-parity, or
leaderboard authority. Only `official_score_evidence.v1` plus measured baseline
coverage can satisfy a confirmed benchmark claim.
