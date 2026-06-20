---
status: in_progress
created: 2026-06-20
---

# Fix failing full test run

## Goal

Restore the full pytest suite by fixing the three observed failures from
`uv run pytest tests/`.

## Tasks

- Re-export the blocked timing sidecar helper from the compatibility wrapper.
- Restore v1.21 concerns wording expected by release documentation tests.
- Re-run the focused failing tests, then rerun the full test suite if focused
  tests pass.
