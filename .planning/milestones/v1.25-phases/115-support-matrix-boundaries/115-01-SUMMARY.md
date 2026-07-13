---
phase: 115
phase_name: "Support Matrix Boundaries"
plan: 1
status: complete
completed_at: "2026-06-01"
requirements: [SUPPORT-01, SUPPORT-02, SUPPORT-03, SUPPORT-04]
---

# Phase 115 Summary

## Completed Work

- Added an engineering prerelease support matrix to `docs/user/rocm.md` covering
  RDNA 4, Docker/container ROCm user-space, MI300X-on-CDNA3, and CDNA4.
- Clarified `docs/user/CLAIMS.md` so prerelease evidence cannot be read as native
  host validation, paper parity, upstream SOLAR parity, score authority,
  leaderboard readiness, hard-sandbox authority, or new hardware validation.
- Updated `docs/internal/release_candidate_validation.md` to point prerelease readers
  back to the support matrix and repeat the concise container, MI300X-on-CDNA3,
  and CDNA4 boundaries.
- Added focused documentation guardrails in
  `tests/sol_execbench/test_research_release_docs.py`.
- Preserved legacy public-contract wording while adding a companion assertion
  that MI300X is the CDNA3 hardware target rather than a separate architecture
  target.

## Evidence

- Implementation commit: `9841dee docs: clarify prerelease support matrix boundaries`
- Review/fix commit: `a48fbbe docs(115): review support matrix boundaries`
- Plan: `.planning/phases/115-support-matrix-boundaries/115-01-PLAN.md`
- Review: `.planning/phases/115-support-matrix-boundaries/115-REVIEW.md`

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_release_candidate_validation.py -q
```

Result: `75 passed in 11.51s`.

## Requirement Closure

- SUPPORT-01: Complete. RDNA 4 engineering-prerelease evidence is visible and
  scoped to recorded artifacts/commands.
- SUPPORT-02: Complete. Docker/container ROCm user-space evidence is explicitly
  not native-host validation.
- SUPPORT-03: Complete. MI300X is named as the concrete CDNA3 hardware target
  represented by `gfx942`; full-suite validation remains deferred without a
  complete real-hardware evidence chain.
- SUPPORT-04: Complete. CDNA4 validation is explicitly unavailable because
  suitable hardware is not currently accessible.
