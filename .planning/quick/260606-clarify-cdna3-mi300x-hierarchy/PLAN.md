# Clarify CDNA3 And MI300X Hierarchy

## Objective

Correct current documentation so CDNA 3 is described as the architecture family
and MI300X/MI308X are sibling GPU models under that family.

## Scope

- Replace slash/or wording that implies CDNA3 and MI300X are peers.
- Keep MI308X evidence bounded as CDNA3/gfx942 infrastructure evidence.
- Keep MI300X validation blocked until exact-hardware evidence exists.
- Update current guardrail tests and prerelease script wording.

## Verification

- Run focused documentation tests.
- Run Ruff on touched Python/test files.
