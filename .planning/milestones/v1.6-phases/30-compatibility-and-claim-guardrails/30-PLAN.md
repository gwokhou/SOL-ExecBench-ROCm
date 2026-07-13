# Phase 30 Plan: Compatibility and Claim Guardrails

**Created:** 2026-05-22
**Status:** Ready for execution
**Requirements:** COMPAT-01, COMPAT-02, COMPAT-03, COMPAT-04, CLAIM-01, CLAIM-02, CLAIM-03

## Goal

Prove public contracts remain compatible and keep CDNA3/NVIDIA equivalence
claims out of v1.6.

## Tasks

### 30-01: Add v1.6 compatibility guardrail tests

**Files:**
- `tests/sol_execbench/test_public_contract_guardrails.py`

**Work:**
- Assert primary `sol-execbench` help does not expose AMD score/profiler options.
- Assert derived artifacts remain separate from canonical trace JSONL.
- Assert dataset runner AMD score option is additive script surface.

### 30-02: Add claim boundary tests and wording fixes

**Files:**
- `src/sol_execbench/core/scoring/amd_score.py`
- `tests/sol_execbench/test_public_contract_guardrails.py`
- `tests/sol_execbench/test_amd_native_score.py`
- `docs/internal/analysis.md`

**Work:**
- Update stale CDNA3 warning text so it is not v1.5-specific.
- Assert v1.6 excludes real CDNA3 `gfx94*` full-suite validation.
- Assert AMD-native reports do not claim NVIDIA B200, upstream SOLAR, or
  leaderboard equivalence.

### 30-03: Verify focused compatibility suite

**Files:**
- `.planning/phases/30-compatibility-and-claim-guardrails/30-VERIFICATION.md`

**Work:**
- Run public contract, AMD SOL, rocprof timing, timing policy, AMD score, and
  dataset score tests.
- Run lint for changed files.

## Verification Commands

- `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_timing_policy.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py`
- `uv run ruff check src/sol_execbench/core/scoring/amd_score.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_native_score.py`

## Requirement Mapping

| Requirement | Task |
|-------------|------|
| COMPAT-01 | 30-01, 30-03 |
| COMPAT-02 | 30-01, 30-03 |
| COMPAT-03 | 30-01, 30-03 |
| COMPAT-04 | 30-03 |
| CLAIM-01 | 30-02 |
| CLAIM-02 | 30-02 |
| CLAIM-03 | 30-02 |
