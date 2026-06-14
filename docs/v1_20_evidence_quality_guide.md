# v1.20 Evidence Quality Guide

v1.20 adds local sidecar reports for reviewing evidence quality before making
stronger benchmark claims. These reports do not change canonical Trace,
Definition, Workload, Solution, correctness, timing, score, or evaluator
semantics.

## Surfaces

| Surface | Script | Purpose |
| --- | --- | --- |
| `sol_execbench.consistency_report.v1` | `scripts/internal/reports/report_consistency.py` | Detect contradictions across closure, denominator, Matrix, runtime/static evidence, AMD score, and AMD bound sanity reports. |
| `sol_execbench.evaluation_stability.v1` | `scripts/internal/reports/report_evaluation_stability.py` | Classify timing quality as stable, noisy, insufficient-samples, missing-timing, clock-unlocked, profiler-overhead-risk, or backend-unsupported. |
| `sol_execbench.claim_upgrade.v1` | `scripts/internal/reports/report_claim_upgrade.py` | Evaluate prerequisites for diagnostic-only, container-validated, native-host-validated, score-authoritative, paper-parity-candidate, and leaderboard-ready claims. |
| `sol_execbench.trust_summary.v1` | `scripts/internal/reports/report_trust_summary.py` | Combine consistency, stability, claim-upgrade, closure, denominator, Matrix, score, and bound status into a concise review artifact. |

## Example Flow

All paths below are demo paths. Keep generated artifacts under a local output
directory and reference source reports by bounded relative refs/checksums.

```bash
UV_CACHE_DIR=out/v1_20_demo/uv-cache uv run scripts/internal/reports/report_consistency.py \
  --execution-closure out/v1_20_demo/execution_closure.json \
  --paper-denominator out/v1_20_demo/paper_denominator.json \
  --matrix-report out/v1_20_demo/matrix.json \
  --amd-score-report out/v1_20_demo/amd_score.json \
  --amd-sol-report out/v1_20_demo/amd_sol.json \
  --solar-derivation out/v1_20_demo/solar_derivation.json \
  --amd-bound-sanity out/v1_20_demo/amd_bound_sanity.json \
  --json-out out/v1_20_demo/consistency.json \
  --markdown-out out/v1_20_demo/consistency.md

UV_CACHE_DIR=out/v1_20_demo/uv-cache uv run scripts/internal/reports/report_evaluation_stability.py \
  --timing-evidence out/v1_20_demo/timing/problem-001.timing.json \
  --json-out out/v1_20_demo/evaluation_stability.json \
  --markdown-out out/v1_20_demo/evaluation_stability.md

UV_CACHE_DIR=out/v1_20_demo/uv-cache uv run scripts/internal/reports/report_claim_upgrade.py \
  --consistency-report out/v1_20_demo/consistency.json \
  --evaluation-stability out/v1_20_demo/evaluation_stability.json \
  --execution-closure out/v1_20_demo/execution_closure.json \
  --paper-denominator out/v1_20_demo/paper_denominator.json \
  --amd-score-report out/v1_20_demo/amd_score.json \
  --amd-sol-report out/v1_20_demo/amd_sol.json \
  --solar-derivation out/v1_20_demo/solar_derivation.json \
  --json-out out/v1_20_demo/claim_upgrade.json \
  --markdown-out out/v1_20_demo/claim_upgrade.md

UV_CACHE_DIR=out/v1_20_demo/uv-cache uv run scripts/internal/reports/report_trust_summary.py \
  --consistency-report out/v1_20_demo/consistency.json \
  --evaluation-stability out/v1_20_demo/evaluation_stability.json \
  --claim-upgrade out/v1_20_demo/claim_upgrade.json \
  --execution-closure out/v1_20_demo/execution_closure.json \
  --paper-denominator out/v1_20_demo/paper_denominator.json \
  --matrix-report out/v1_20_demo/matrix.json \
  --amd-score-report out/v1_20_demo/amd_score.json \
  --amd-sol-report out/v1_20_demo/amd_sol.json \
  --solar-derivation out/v1_20_demo/solar_derivation.json \
  --amd-bound-sanity out/v1_20_demo/amd_bound_sanity.json \
  --json-out out/v1_20_demo/trust_summary.json \
  --markdown-out out/v1_20_demo/trust_summary.md
```

## Claim Boundary

v1.20 does not add full 235-problem paper validation, CDNA3-family validation, including MI300X (`gfx942`), CDNA4 validation, native-host Matrix authority, hosted leaderboard readiness, or upstream SOLAR parity.
It is not paper validation, not paper parity, not leaderboard authority,
not native-host validation, and not new-hardware validation.

Consistency lint is diagnostic-only. Stability supports interpretation only and
does not create correctness, timing, score, paper-parity, native-host, or
leaderboard authority. Claim-upgrade reports evaluate prerequisites only and do
not mutate source authority fields. Trust summaries are review guidance only.

## Fixtures

Demo-only fixtures live under `docs/examples/v1_20_evidence_quality/`. They show
consistent, contradictory, noisy, and claim-blocked shapes with synthetic
checksums, bounded relative refs, and negative claim wording.
