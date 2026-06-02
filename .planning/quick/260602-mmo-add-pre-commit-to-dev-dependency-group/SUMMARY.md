---
quick_id: 260602-mmo
slug: add-pre-commit-to-dev-dependency-group
status: complete
completed_at: "2026-06-02T08:20:00Z"
---

# Quick Task 260602-mmo: Add pre-commit To Dev Dependencies

## Result

Added `pre-commit` to the `dev` dependency group so the default hook workflow is
available after `uv sync --all-groups`.

## Changes

- Added `pre-commit>=4.6.0` to `[dependency-groups].dev`.
- Updated `uv.lock` with `pre-commit` and its transitive dependencies.
- Updated hook setup docs to use `uv run pre-commit install`.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv lock` - passed.
- `PRE_COMMIT_HOME=/tmp/pre-commit-cache UV_CACHE_DIR=/tmp/uv-cache uv run pre-commit validate-config` - passed.
- `PRE_COMMIT_HOME=/tmp/pre-commit-cache UV_CACHE_DIR=/tmp/uv-cache uv run pre-commit run --hook-stage pre-push --all-files` - Ty Type Check passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check` - passed.
