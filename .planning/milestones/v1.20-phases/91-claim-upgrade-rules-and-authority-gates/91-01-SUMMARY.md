---
status: complete
requirements-completed: [CLAIM-01, CLAIM-02, CLAIM-03, CLAIM-04]
---

# Phase 91 Plan 01 Summary: Claim Rule Contract

**Status:** Complete
**Completed:** 2026-05-31

## Delivered

- Added `src/sol_execbench/core/claim_upgrade.py` with strict `sol_execbench.claim_upgrade.v1` report models.
- Defined claim levels for diagnostic-only, container-validated, native-host-validated, score-authoritative, paper-parity-candidate, and leaderboard-ready states.
- Implemented prerequisite checks against consistency, stability, closure, denominator, Matrix, score, bound sanity, and hardware validation evidence.
- Added deterministic checksums, JSON serialization, Markdown rendering, unmet prerequisites, blockers, and next-evidence hints.
- Kept the report prerequisite-only and non-mutating.

## Requirements Covered

- CLAIM-01
- CLAIM-02
- CLAIM-03
- CLAIM-04

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_claim_upgrade.py tests/sol_execbench/test_claim_upgrade_script.py tests/sol_execbench/test_public_contract_guardrails.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/claim_upgrade.py scripts/report_claim_upgrade.py tests/sol_execbench/test_claim_upgrade.py tests/sol_execbench/test_claim_upgrade_script.py tests/sol_execbench/test_public_contract_guardrails.py`
