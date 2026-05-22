# Phase 26 Context: AMD-native Scoring and Guarded Reports

**Status:** Ready for planning  
**Created:** 2026-05-22  
**Requirements:** SCORE-01, SCORE-02, SCORE-03, SCORE-04, COMPAT-01,
COMPAT-02, CLAIM-01, CLAIM-02

## Objective

Combine measured trace timing with Phase 25 AMD SOL bound artifacts to produce
derived AMD-native score reports. Preserve existing public contracts and keep
CDNA3 validation explicitly excluded from v1.5.

## User Constraints

- The milestone does not include CDNA3 validation.
- Scores must not claim NVIDIA B200, SOLAR leaderboard, or cross-vendor
  equivalence.
- Derived SOL and timing evidence must remain outside canonical trace JSONL.
- Existing CLI behavior, solution schema, eval-driver correctness semantics,
  and reward-hack defenses must not regress.

## Current Inputs

- `src/sol_execbench/sol_score.py` preserves the original SOL-Score formula.
- `src/sol_execbench/core/scoring/amd_sol.py` produces AMD SOL bound artifacts.
- `src/sol_execbench/core/baseline.py` provides baseline-relative comparison.
- `src/sol_execbench/core/reporting.py` already models derived evidence as
  noncanonical output.
- `src/sol_execbench/core/scoring_guardrails.py` contains AMD claim warnings.

## Decision

Implement Phase 26 as a pure derived-report layer, not as a mutation to trace
serialization or the primary `sol-execbench` CLI. This is enough to satisfy the
phase requirements and keeps public compatibility intact.
