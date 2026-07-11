### Task 2: Architecture adapters and HIP/ROCm backend orchestration

**Files:**
- Create: `src/sol_execbench/core/scoring/hardware_calibration/environment.py`
- Create: `src/sol_execbench/core/scoring/hardware_calibration/hip_probe.py`
- Create: `src/sol_execbench/core/scoring/hardware_calibration/builder.py`
- Test: `tests/sol_execbench/core/scoring/hardware_calibration/test_environment.py`
- Test: `tests/sol_execbench/core/scoring/hardware_calibration/test_hip_probe.py`
- Test: `tests/sol_execbench/core/scoring/hardware_calibration/test_builder.py`

**Interfaces:** `discover_gpu(device) -> GpuEnvironment`, `adapter_for(architecture) -> ArchitectureAdapter`, and `run_calibration(request) -> HardwareCalibrationArtifact`. Each adapter declares `CalibrationProfileKey(kind, operation, input_dtype, output_dtype, path)` candidates.

- [ ] **Step 1: Write failing tests** — Parameterize `gfx1200`, `gfx942`, and `gfx950`, asserting every adapter declares portable FP32 vector and streaming-copy candidates. Assert a missing clock adapter returns `collection_status="collected"` and `validation_status="provisional"`, while `--require-clock-lock` rejects it.

- [ ] **Step 2: Verify red** — Run `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_environment.py tests/sol_execbench/core/scoring/hardware_calibration/test_hip_probe.py tests/sol_execbench/core/scoring/hardware_calibration/test_builder.py -n 0 -v`. Expected: FAIL because adapters and probes do not exist.

- [ ] **Step 3: Implement minimal adapters** — Route `gfx12*`, `gfx94*`, and `gfx95*` from runtime discovery. Each adapter compiles, executes, and numerically checks portable vector/memory and declared ISA candidates. Explicit unsupported probe results are `unavailable`; absent or failed probes are `unknown`. Reuse `clock_lock.lock_clocks()` / `unlock_clocks()` on RDNA4. When the profiler backend works, run `profile --bench-only`, checksum its raw CSV, and map only recognised metrics to the same matrix keys.

- [ ] **Step 4: Verify green and commit** — Run the Step 2 command; expected PASS. Commit: `git add src/sol_execbench/core/scoring/hardware_calibration tests/sol_execbench/core/scoring/hardware_calibration && git commit -s -m "Add portable AMD calibration backends"`.

