---
quick_id: 260602-mbt
slug: add-git-pre-push-ty-check-hook
status: in_progress
created_at: "2026-06-02T08:04:35.879Z"
---

# Quick Task 260602-mbt: Add Git Pre-Push Ty Hook

## Goal

Add a git push hook that runs `uv run ty check` before pushing.

## Plan

1. Extend `.pre-commit-config.yaml` with a local `pre-push` Ty hook.
2. Update contributor/development setup docs to install the `pre-push` hook.
3. Verify the hook command and pre-commit configuration.
