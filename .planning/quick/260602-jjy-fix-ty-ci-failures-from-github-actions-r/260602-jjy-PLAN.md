---
quick_id: 260602-jjy
slug: fix-ty-ci-failures-from-github-actions-r
status: in_progress
created_at: "2026-06-02T06:04:44.195Z"
---

# Quick Task 260602-jjy: Fix Ty CI failures

## Goal

Fix the `uv run ty check` failures reported by GitHub Actions run
26798761166, job 79000755601, without changing runtime benchmark semantics.

## Plan

1. Reproduce the Ty failures locally.
2. Apply focused typing fixes for dynamic monkeypatching, test payload helpers,
   TOML parsing, and nullable model fields.
3. Run `uv run ty check`, focused tests for touched files, and Ruff.
4. Record the outcome in a quick task summary.
