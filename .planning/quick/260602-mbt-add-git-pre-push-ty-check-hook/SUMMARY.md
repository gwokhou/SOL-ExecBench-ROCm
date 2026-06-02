---
quick_id: 260602-mbt
slug: add-git-pre-push-ty-check-hook
status: complete
completed_at: "2026-06-02T08:07:00Z"
---

# Quick Task 260602-mbt: Add Git Pre-Push Ty Hook

## Result

Added a local pre-commit-managed `pre-push` hook named `ty-check` that runs
`uv run ty check` before `git push`.

## Changes

- Added `ty-check` to `.pre-commit-config.yaml` with `stages: [pre-push]` and
  `pass_filenames: false`.
- Updated `docs/DEVELOPMENT.md` and `CONTRIBUTING.md` with the
  `pre-commit install --hook-type pre-push` setup command.
- Installed the hook locally at `.git/hooks/pre-push`.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check` - passed.
- `PRE_COMMIT_HOME=/tmp/pre-commit-cache UV_CACHE_DIR=/tmp/uv-cache uv run --with pre-commit pre-commit validate-config` - passed.
- `PRE_COMMIT_HOME=/tmp/pre-commit-cache UV_CACHE_DIR=/tmp/uv-cache uv run --with pre-commit pre-commit run ty-check --hook-stage pre-push --all-files` - passed.
