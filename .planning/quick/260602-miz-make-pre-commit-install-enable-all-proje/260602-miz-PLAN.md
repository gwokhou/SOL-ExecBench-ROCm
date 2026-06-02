---
quick_id: 260602-miz
slug: make-pre-commit-install-enable-all-proje
status: in_progress
created_at: "2026-06-02T08:13:11.522Z"
---

# Quick Task 260602-miz: Default-Enable Project Hooks

## Goal

Make the standard `pre-commit install` command install the project's
pre-commit, commit-message, and pre-push hooks by default.

## Plan

1. Configure default install hook types in `.pre-commit-config.yaml`.
2. Restrict Ruff hooks to the `pre-commit` stage so pushes only run Ty.
3. Update docs to use one default install command instead of separate hook-type
   installs.
4. Validate the pre-commit config and hook behavior.
