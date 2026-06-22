# Quick Task 260622-eaa: Fix pre-push ty-check blockers before pushing local main

## Task

Fix the local `ty-check` pre-push failures that blocked `git push origin main`, then rerun the relevant checks and push the already committed local `main` history.

## Plan

1. Fix type narrowing in CLI/profile-summary/static-evidence helpers without changing runtime behavior.
2. Fix narrow static typing issues in feedback aggregation and test fixtures.
3. Run `uv run ty check`, targeted tests for touched areas, commit the fix and GSD artifacts, then retry `git push origin main`.
