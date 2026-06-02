---
quick_id: 260602-miz
slug: make-pre-commit-install-enable-all-proje
status: complete
completed_at: "2026-06-02T08:15:00Z"
---

# Quick Task 260602-miz: Default-Enable Project Hooks

## Result

Configured `pre-commit install` to install all project hook types by default:
`pre-commit`, `commit-msg`, and `pre-push`.

## Changes

- Added `default_install_hook_types: [pre-commit, commit-msg, pre-push]`.
- Restricted Ruff and Ruff format hooks to the `pre-commit` stage.
- Kept the Ty hook on the `pre-push` stage.
- Updated setup docs to use a single `pre-commit install` command.
- Reinstalled local hooks at `.git/hooks/pre-commit`, `.git/hooks/commit-msg`,
  and `.git/hooks/pre-push`.

## Verification

- `pre-commit validate-config` - passed.
- `pre-commit run --hook-stage pre-push --all-files` - Ty Type Check passed.
