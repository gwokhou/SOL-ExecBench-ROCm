# Phase 26 Summary: AMD-native Scoring and Guarded Reports

**Status:** Complete  
**Completed:** 2026-05-22  
**Code commit:** `8a1ce2e`  
**Requirements:** SCORE-01, SCORE-02, SCORE-03, SCORE-04, COMPAT-01,
COMPAT-02, CLAIM-01, CLAIM-02

## Delivered

- Added `sol_execbench.core.scoring.amd_score` for derived AMD-native
  per-workload and suite score reports.
- Reused the existing `sol_score()` formula without changing compatibility
  behavior.
- Added evidence references for timing and AMD SOL bound artifacts.
- Added guardrails for unsupported SOL evidence, incomplete numeric inputs,
  unvalidated hardware models, and CDNA3 no-validation claims.
- Documented AMD-native score reports as derived artifacts that do not mutate
  canonical trace JSONL and do not claim NVIDIA B200/SOLAR/leaderboard
  equivalence.

## Notes

- The report layer is pure Python and hardware-independent.
- CDNA3 `gfx94*` model scaffolding remains unvalidated and guarded, matching
  the user constraint for v1.5.
