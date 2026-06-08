---
status: complete
resolved_at: "2026-06-08"
outcome: superseded
---

# Remote CI Failure Debug

## Symptom

Remote GitHub Actions CI is reported failing. Local `gh` authentication is invalid,
so the first local fallback is to run the same workflow commands from
`.github/workflows/code-quality.yml`.

## Workflow Commands

- `uv run ruff check .`
- `uv run ty check`
- `uv run pytest tests/sol_execbench`
- `uv run pytest tests/examples/test_examples.py -k consistency`

## Notes

- Current branch: `main`
- Open PRs found by GitHub connector: none
- `gh auth status` reports an invalid token for `gwokhou`.

## Resolution

Closed during v1.31 milestone completion artifact hygiene. The session did not
identify an active PR or current remote CI blocker to fix, and later local
quality gates were rerun during v1.30/v1.31 milestone work. Any future remote
CI failure should be opened as a fresh debug session with current `gh` or
GitHub Actions evidence.
