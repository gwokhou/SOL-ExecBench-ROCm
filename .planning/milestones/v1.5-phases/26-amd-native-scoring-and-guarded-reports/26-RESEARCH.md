# Phase 26 Research: AMD-native Scoring and Guarded Reports

**Status:** Complete  
**Date:** 2026-05-22

## Existing Patterns

| Area | Existing implementation | Reuse |
|------|-------------------------|-------|
| Formula | `sol_execbench.sol_score.sol_score()` | Use unchanged for per-problem score math. |
| AMD SOL bound evidence | `core/scoring/amd_sol.py` | Require a bound artifact before scoring. |
| Baseline comparison | `core/baseline.py` | Reference baseline-relative results without changing semantics. |
| Guardrails | `core/scoring_guardrails.py` | Reuse warning style and add evidence-specific claim statuses. |
| Derived reports | `core/reporting.py` | Keep AMD scoring reports derived and noncanonical. |

## Implementation Shape

Add `core/scoring/amd_score.py` with pure dataclasses:

- `AmdNativeScore`: one workload score with measured latency, baseline latency,
  SOL bound, score value, claim level, guardrail warnings, and evidence refs.
- `AmdNativeSuiteReport`: aggregate report with per-workload scores, mean score,
  schema version, derived marker, canonical output marker, and optional baseline
  summary.

The report should reject or guard incomplete evidence:

- missing or nonpositive measured latency: unsupported score
- missing or nonpositive SOL bound: unsupported score
- unvalidated CDNA3 hardware model: score can be computed, but report must carry
  no-CDNA3-validation warning
- unsupported operation estimates: report must include unsupported evidence
  warning

## Non-Goals

- Do not add fields to trace JSONL.
- Do not change `sol_score()`.
- Do not claim NVIDIA B200/SOLAR equivalence.
- Do not promote `gfx94*` validation status in v1.5.
