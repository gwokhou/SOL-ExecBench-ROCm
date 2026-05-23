---
phase: 47-derivation-contract-and-golden-fixture-matrix
plan: 06
subsystem: tests
tags: [solar, fixture-matrix, public-contract, guardrails]

requires:
  - phase: 47-01
    provides: SOLAR derivation contract
  - phase: 47-02
    provides: fixture loader
  - phase: 47-03
    provides: attention and MoE fixtures
  - phase: 47-04
    provides: convolution and SSM/Mamba fixtures
  - phase: 47-05
    provides: embedding/positional and linear projection fixtures
provides:
  - Full fixture matrix coverage tests
  - Phase 47 public contract and claim-boundary guardrails
affects: [phase-47, phase-48, phase-49, phase-50, phase-51, phase-52]
requirements-completed: [TEST-01, TEST-02]
duration: 3min
completed: 2026-05-23
---

# Phase 47 Plan 06: Fixture Matrix And Public Guardrails Summary

Added full fixture matrix tests and public contract guardrails for Phase 47.

## Commits

- `34ca370 test(47-06): add SOLAR fixture matrix guardrails`

## Accomplishments

- Verified the fixture matrix contains all six required families and the positive/degraded/unsupported-or-negative classes.
- Verified degraded and negative fixtures cover `dynamic`, `partial`, `unsupported`, `taxonomy_only`, and `missing_metadata`.
- Added public guardrails ensuring the v1.10 SOLAR derivation contract remains sidecar-only and does not imply paper-scale extraction, hosted leaderboard readiness, NVIDIA Blackwell/B200 equivalence, or new real-hardware validation.
- Added canonical schema and primary CLI help checks to ensure Phase 47 fields/options did not leak into public runtime contracts.

## Verification

Passed:

```bash
uv run pytest tests/sol_execbench/test_solar_derivation_contract.py -n 0 -x
```

Result: `11 passed`.

Passed:

```bash
uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -n 0 -x
```

Result: `16 passed`.

Passed:

```bash
uv run pytest tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0
```

Result: `27 passed`.

## Notes

- Updated the internal contract with explicit no-claim guardrail phrases for exact test matching.
- Preserved the legacy `CDNA 3 / MI300X real-hardware validation` wording in requirements so existing public contract guardrails remain active.

## Self-Check: PASSED

- TEST-01 covered by the full family/class matrix.
- TEST-02 covered by negative/degradation categories and expectation assertions.
- No production scoring, extraction, modeling, schema, or primary CLI code changed.

---
*Phase: 47-derivation-contract-and-golden-fixture-matrix*
*Completed: 2026-05-23*
