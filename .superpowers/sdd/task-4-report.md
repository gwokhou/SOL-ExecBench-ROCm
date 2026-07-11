# Task 4 Report: Unified official-score CLI policy

## Delivered

- Enforced the sole `fixed_suite_denominator_zero_for_blocked` policy through
  a case-sensitive `click.Choice`; `--aggregation-policy` remains explicitly
  required and no legacy free-text example remains in CLI help.
- Updated CLI coverage to use `OFFICIAL_AGGREGATION_POLICY`, reject the legacy
  policy while displaying the allowed value, retain missing-policy rejection,
  and verify a blocked placeholder contributes zero to a two-workload suite
  (`0.75` becomes `0.375`).
- Updated all confirmed-evidence fixtures to Task 1's serialized suite shape:
  fixed policy, total/blocked/zero-scored counts, and `0.0` score for
  all-blocked non-empty suites.
- Added fixture assertions for the policy, fixed denominator, and serialized
  count invariants.

## TDD evidence

1. The legacy-policy CLI regression test initially failed because the CLI
   accepted arbitrary non-empty policy text.
2. The fixture serialization regressions initially failed because all bundles
   still carried the legacy policy and omitted the new suite fields.
3. The implementation and fixture updates made the focused suite green.

## Verification

```text
uv run pytest tests/sol_execbench/cli/commands/test_official_score_cli.py tests/sol_execbench/core/evidence/test_confirmed_evidence_fixtures.py -q
29 passed in 2.23s

uv run --with ruff ruff check src/sol_execbench/cli/commands/official_score.py tests/sol_execbench/cli/commands/test_official_score_cli.py tests/sol_execbench/core/evidence/test_confirmed_evidence_fixtures.py
All checks passed!

git diff --check
exit 0
```

## Scope

Only Task 4 CLI, CLI tests, confirmed-evidence fixtures, fixture tests, and
this report are intended for the Task 4 commit. Pre-existing concurrent edits
to `src/sol_execbench/core/dataset/runner_scoring.py` and
`tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py` were left
untouched and excluded.
