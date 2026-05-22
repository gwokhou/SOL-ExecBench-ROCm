# Phase 26 Verification

**Status:** Passed  
**Date:** 2026-05-22

## Commands

```bash
uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_baseline_comparison.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/scoring/amd_score.py tests/sol_execbench/test_amd_native_score.py
```

## Results

- Full Phase 26 suite: `26 passed`
- Focused Phase 26 suite after final type annotation adjustment: `5 passed`
- `ruff`: all checks passed

## Requirement Evidence

- SCORE-01: `score_amd_native_workload()` computes per-workload scores from
  measured timing and AMD SOL bound artifacts.
- SCORE-02: baseline comparison remains baseline-relative; AMD score docs
  explicitly reject NVIDIA B200, SOLAR, and leaderboard equivalence claims.
- SCORE-03: `AmdNativeSuiteReport` preserves per-workload timing and SOL-bound
  evidence references.
- SCORE-04: reports carry guardrails for unsupported, incomplete, and
  unvalidated evidence.
- COMPAT-01/COMPAT-02: public contract tests pass and reports remain derived
  artifacts with `canonical_output=trace_jsonl`.
- CLAIM-01/CLAIM-02: CDNA3 full-suite validation remains excluded and `gfx94*`
  reports carry a no-validation warning.
