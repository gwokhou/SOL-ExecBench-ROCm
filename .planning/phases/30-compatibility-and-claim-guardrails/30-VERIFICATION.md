---
status: passed
phase: 30
---

# Phase 30 Verification: Compatibility and Claim Guardrails

**Verified:** 2026-05-22
**Result:** Passed

## Must-Haves

| Requirement | Result | Evidence |
|-------------|--------|----------|
| COMPAT-01 | Passed | Public contract tests cover trace JSONL fields/parsing and v1.6 derived artifact separation. |
| COMPAT-02 | Passed | Existing public schema guardrails continue to pass for solution, workload, and trace models. |
| COMPAT-03 | Passed | Primary `sol-execbench` help does not expose v1.6 derived workflow options; dataset report option is additive script surface. |
| COMPAT-04 | Passed | Focused compatibility suite covers all v1.6 analyzer, timing, scoring, and public contract changes. |
| CLAIM-01 | Passed | Docs/tests state real CDNA3 `gfx94*` full-suite validation is outside v1.6. |
| CLAIM-02 | Passed | Docs/tests preserve no NVIDIA B200, upstream SOLAR, or leaderboard equivalence claim. |
| CLAIM-03 | Passed | Tests verify hardware model source, confidence, validation status, and score evidence refs survive artifacts. |

## Commands

```bash
uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py
uv run ruff check src/sol_execbench/core/scoring/amd_score.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_native_score.py
```

Both commands passed.

## Residual Risk

This phase ran the focused compatibility and v1.6 regression set, not the full
repository test suite.
