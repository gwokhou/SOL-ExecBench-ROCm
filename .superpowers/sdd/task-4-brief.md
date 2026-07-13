### Task 4: Official authority gate and CLI

**Files:**
- Modify: `src/sol_execbench/core/scoring/amd_score/models.py`
- Modify: `src/sol_execbench/core/scoring/amd_score/workload.py`
- Modify: `src/sol_execbench/core/scoring/official_score.py`
- Create: `src/sol_execbench/cli/commands/hardware_model.py`
- Modify: `src/sol_execbench/cli/commands/root.py`
- Modify: `docs/user/CONFIGURATION.md`
- Modify: `docs/user/TESTING.md`
- Modify: `docs/internal/sol_score_gap_and_amd_reuse_report.md`
- Test: `tests/sol_execbench/core/scoring/test_amd_native_score.py`
- Test: `tests/sol_execbench/core/evidence/test_official_score_evidence.py`
- Test: `tests/sol_execbench/cli/commands/test_hardware_model_cli.py`

**Interfaces:** Add `BoundEligibilityEvidence(amd_sol_status, solar_status, hardware_profile_state, hardware_validation_status, model_validation_status, warnings)` to serialized AMD scores. Add `sol-execbench hardware-model calibrate --device 0 --output calibration.json [--architecture GFX] [--require-clock-lock] [--offline] [--no-auto-install]` and `hardware-model build --calibration calibration.json --output calibrated-model.json`.

- [ ] **Step 1: Write failing tests** — A legacy score without `BoundEligibilityEvidence` receives `missing_bound_eligibility` and cannot become official. Exact validated profile evidence passes. CLI writes a rejected artifact before reporting nonzero exit.

- [ ] **Step 2: Verify red** — Run `uv run pytest tests/sol_execbench/core/scoring/test_amd_native_score.py tests/sol_execbench/core/evidence/test_official_score_evidence.py tests/sol_execbench/cli/commands/test_hardware_model_cli.py -n 0 -v`. Expected: FAIL because score evidence and command are absent.

- [ ] **Step 3: Implement propagation and commands** — Derive eligibility from AMD SOL aggregate, SOLAR aggregate, hardware profile, validation states, and warnings. Block `missing_bound_eligibility`, non-scored aggregates, non-supported profiles, non-validated model, and degraded/inexact/unsupported warnings. Add lazy CLI dispatch; write JSON before `ClickException`. Document managed dependencies, external-model selection, and diagnostic versus authoritative results.

- [ ] **Step 4: Verify, run live check, and commit** — Run `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration tests/sol_execbench/core/scoring/test_amd_hardware_models.py tests/sol_execbench/core/scoring/test_amd_sol_v2.py tests/sol_execbench/core/scoring/test_amd_native_score.py tests/sol_execbench/core/evidence/test_official_score_evidence.py tests/sol_execbench/cli/commands/test_hardware_model_cli.py -n 0 -v && uv run --with ruff ruff check .`; expected PASS. Then run the new marker-gated test with `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_live_calibration.py -m requires_rdna4 -n 0 -v`; inspect that profiler gaps are `unknown`. Commit: `git add src tests docs && git commit -s -m "Add calibrated hardware model workflow"`.
