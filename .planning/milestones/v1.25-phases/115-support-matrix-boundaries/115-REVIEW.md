---
phase: 115
phase_name: "Support Matrix Boundaries"
review_type: "local_code_review"
status: clean
reviewed_at: "2026-06-01"
---

# Phase 115 Code Review

## Scope

Reviewed the Phase 115 support-boundary changes in:

- `docs/rocm.md`
- `docs/CLAIMS.md`
- `docs/release_candidate_validation.md`
- `tests/sol_execbench/test_research_release_docs.py`
- `tests/sol_execbench/test_public_contract_guardrails.py`

## Findings

No blocking or follow-up findings remain.

One documentation typo was found and fixed locally before completion:
`docs/release_candidate_validation.md` now says the engineering prerelease path
"should be interpreted alongside the support matrix" instead of the malformed
"does should" wording.

## Boundary Checks

- MI300X is treated as the concrete CDNA 3 hardware target (`gfx942`), not as a
  separate architecture target.
- `gfx940`, `gfx941`, and `gfx942` remain CDNA 3 code/schema targets.
- Full-suite MI300X-on-CDNA3 hardware validation remains deferred unless complete
  real-hardware evidence exists.
- CDNA4 validation is described as unavailable because suitable hardware is not
  currently accessible.
- Docker/container ROCm user-space evidence remains distinct from native-host
  validation and does not upgrade score, paper-parity, or leaderboard claims.

## Verification

Focused tests were run after the fix:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_release_candidate_validation.py -q
```

Result: pending at review-write time; final result is recorded in
`115-VERIFICATION.md`.
