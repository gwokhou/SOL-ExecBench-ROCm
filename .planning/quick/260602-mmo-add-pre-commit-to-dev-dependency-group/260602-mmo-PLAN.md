---
quick_id: 260602-mmo
slug: add-pre-commit-to-dev-dependency-group
status: in_progress
created_at: "2026-06-02T08:17:36.972Z"
---

# Quick Task 260602-mmo: Add pre-commit To Dev Dependencies

## Goal

Make the dependency groups complete for the default hook workflow by including
`pre-commit` in the development dependency group.

## Plan

1. Add `pre-commit` to `[dependency-groups].dev` in `pyproject.toml`.
2. Update `uv.lock`.
3. Validate `uv run pre-commit validate-config`, `uv run ty check`, and
   pre-push hook behavior.
