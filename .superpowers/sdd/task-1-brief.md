### Task 1: Calibration contract and profiler environment

**Files:**
- Create: `src/sol_execbench/core/scoring/hardware_calibration/__init__.py`
- Create: `src/sol_execbench/core/scoring/hardware_calibration/models.py`
- Create: `src/sol_execbench/core/scoring/hardware_calibration/statistics.py`
- Create: `src/sol_execbench/core/scoring/hardware_calibration/rocprof_compute.py`
- Test: `tests/sol_execbench/core/scoring/hardware_calibration/test_models.py`
- Test: `tests/sol_execbench/core/scoring/hardware_calibration/test_statistics.py`
- Test: `tests/sol_execbench/core/scoring/hardware_calibration/test_rocprof_compute.py`

**Interfaces:** `CalibrationCandidate(key, state, value, unit, samples, reason_code)`, `HardwareCalibrationArtifact`, `select_conservative_value(samples)`, `ensure_profiler_environment(discovery, offline, auto_install)`, and `run_rocprof_compute_bench_only()`.

- [ ] **Step 1: Write failing tests** — test that an `unknown` candidate with a value raises `ValueError`; seven valid samples retain the top three and select their minimum; managed invocation sets `VIRTUAL_ENV`, prepends `venv/bin` to `PATH`, and sets `PYTHONNOUSERSITE=1`; offline missing dependencies returns `unknown` with `rocprof_compute_dependencies_unavailable_offline`.

- [ ] **Step 2: Verify red** — Run `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_models.py tests/sol_execbench/core/scoring/hardware_calibration/test_statistics.py tests/sol_execbench/core/scoring/hardware_calibration/test_rocprof_compute.py -n 0 -v`. Expected: FAIL because calibration modules do not exist.

- [ ] **Step 3: Implement minimal contract** — Strictly parse/serialize `sol_execbench.hardware_calibration.v1`; require finite positive samples and retained spread `<= 0.05`. Key a file-locked virtual environment by tool version, requirements SHA-256, and interpreter ABI. On explicit calibration only, run `uv venv <venv>` and `uv pip install --python <venv>/bin/python -r <requirements>`. Store the installed-distribution manifest; parse missing/unrecognised roofline metrics as `unknown`.

- [ ] **Step 4: Verify green and commit** — Run the Step 2 command; expected PASS without network. Commit: `git add src/sol_execbench/core/scoring/hardware_calibration tests/sol_execbench/core/scoring/hardware_calibration && git commit -s -m "Add calibration evidence contracts"`.
