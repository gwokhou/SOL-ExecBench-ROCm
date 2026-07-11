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
Verification: the exact `-m requires_rdna4` live test passed on this host, and
the CLI regression plus calibration builder tests passed (9 tests total).

## Follow-up: live calibration outcome and clock-state investigation

The live RDNA4 calibration test intentionally remains strict: it expects the
diagnostic rejection path and does not branch on a successful outcome.  A
requested investigation found no basis for weakening it.  Immediately before
the exact live test, `rocm-smi --showperflevel` reported GPU 0 at
`Performance Level: auto`; the marker-gated test then passed.  The focused
calibration builder and CLI suite uses injected/mocked clock controllers for
its lifecycle tests, passed 12 tests, and the same `rocm-smi` check reported
`auto` both before and after it.  `run_calibration()` also invokes the
controller's `unlock()` unconditionally in its `finally` block.

Therefore STABLE_PEAK was neither a pre-existing host state nor leaked by the
relevant test suite in this reproduction.  No production authority semantics
or test assertion were changed.

Verification:

- `rocm-smi --showperflevel` before/after focused suite — GPU 0 `auto`.
- `.../.venv/bin/python -m pytest tests/sol_execbench/core/scoring/hardware_calibration/test_live_calibration.py::test_live_offline_calibration_writes_rdna4_evidence -m requires_rdna4 -q` — 1 passed in 8.45s.
- `.../.venv/bin/python -m pytest tests/sol_execbench/core/scoring/hardware_calibration/test_builder.py tests/sol_execbench/cli/commands/test_hardware_model_cli.py -q` — 12 passed in 2.27s.
