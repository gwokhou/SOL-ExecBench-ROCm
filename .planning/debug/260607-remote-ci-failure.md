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
