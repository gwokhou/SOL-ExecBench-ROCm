---
status: complete
created: 2026-05-25
completed: 2026-05-25
---

# Fix Ty S0/S1 Diagnostics

## Goal

Reduce Ty diagnostics by fixing safe S0/S1 issues without changing public
runtime semantics.

## Scope

- Test-only raw Pydantic dict construction fixes using helper casts.
- Local payload and JSON typing casts where Ty infers `object`.
- Local narrowing or guards that preserve existing behavior.
- Do not add broad `ty: ignore` suppressions.
- Do not change public schemas or scoring formulas as part of this quick task.

## Verification

- `uv run ty check`
- Focused pytest for touched tests
- `uv run ruff check` on touched files
