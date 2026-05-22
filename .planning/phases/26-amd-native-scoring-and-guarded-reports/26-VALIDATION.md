# Phase 26 Validation Strategy

**Status:** Complete  
**Date:** 2026-05-22

## Required Checks

```bash
uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_baseline_comparison.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/scoring/amd_score.py tests/sol_execbench/test_amd_native_score.py
```

## Acceptance Coverage

- SCORE-01: per-problem AMD-native scores are generated from measured timing
  and AMD SOL bound artifacts.
- SCORE-02: baseline comparison remains baseline-relative and does not claim
  NVIDIA B200, SOLAR, or leaderboard equivalence.
- SCORE-03: suite aggregation preserves workload timing and SOL-bound evidence
  references.
- SCORE-04: reports include guardrails for unsupported, incomplete, and
  unvalidated evidence.
- COMPAT-01: existing public contract tests continue to pass.
- COMPAT-02: AMD scoring report is derived and marks canonical output as
  `trace_jsonl`.
- CLAIM-01/CLAIM-02: CDNA3 remains explicitly unvalidated in score reports.
