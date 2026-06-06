# Clarify MI308X CDNA3 Validation Evidence

## Objective

Correct current CDNA3 validation documentation so recorded cloud evidence is
attributed to MI308X (`gfx942`) hardware, not MI300X. Keep MI300X as a distinct
benchmark-grade validation target even though it shares the `gfx942` code path.

## Scope

- Update current planning handoff/status docs.
- Update public support, claims, release, and research-preview docs.
- Update documentation guardrail tests that assert the CDNA3/MI300X boundary.

## Out Of Scope

- Rewriting historical milestone archives.
- Changing benchmark behavior.
- Changing hardware detection or scoring logic.

## Verification

- Run focused documentation test suites.
- Run Ruff on touched tests.
