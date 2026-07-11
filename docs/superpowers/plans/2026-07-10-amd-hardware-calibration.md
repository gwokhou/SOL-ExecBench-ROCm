# AMD Hardware Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver portable AMD capability calibration, managed ROCm Compute Profiler dependencies, path-specific hardware models, and official-score authority gates.

**Architecture:** Calibration artifacts, backend execution, model resolution, and authority gating are separate. HIP probes provide the portable baseline; ROCm Compute Profiler imports empirical roofline data only when its isolated environment works. Exact operation/dtype/path profiles replace scalar fallback values.

**Tech Stack:** Python 3.12, Click, dataclasses, `uv`, HIP/hipcc, ROCm Compute Profiler, Pytest, Ruff.

## Global Constraints

- Support RDNA4 `gfx12*`, CDNA3 `gfx94*`, and future CDNA4 `gfx95*`.
- Candidate states are exactly `measured`, `unavailable`, or `unknown`; only `measured` has a positive value.
- Generated models are external; packaged models remain provisional.
- Profiler environments live only under ignored `.artifacts/rocprof-compute/`.
- Profiler invocation prepends its managed `bin` to `PATH`, sets `VIRTUAL_ENV`, and sets `PYTHONNOUSERSITE=1`.
- `--offline` and `--no-auto-install` never download; a missing profiler backend is `unknown` while HIP probes continue.
- Preserve v2 hardware-model and v1 score readers; missing new evidence blocks official authority.

---

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

### Task 3: Hardware model v3 and exact SOL profile resolution

**Files:**
- Modify: `src/sol_execbench/core/scoring/amd_hardware_models.py`
- Modify: `src/sol_execbench/core/scoring/amd_bound_estimate/models.py`
- Modify: `src/sol_execbench/core/scoring/amd_bound_estimate/impl.py`
- Modify: `src/sol_execbench/core/scoring/amd_sol/v2_math.py`
- Modify: `src/sol_execbench/core/scoring/hardware_calibration/builder.py`
- Test: `tests/sol_execbench/core/scoring/test_amd_hardware_models.py`
- Test: `tests/sol_execbench/core/scoring/test_amd_sol_v2.py`

**Interfaces:** `HardwareProfile(key, state, value, confidence, evidence_ref)`, `AmdHardwareModel.resolve_compute(operation, input_dtype, output_dtype, path)`, and `AmdHardwareModel.resolve_memory(access)`.

- [ ] **Step 1: Write failing tests** — A v3 BF16 MFMA profile must resolve for a BF16 matrix estimate. A v3 model with only FP32 must not service BF16: the bound is `degraded` and includes `unknown_hardware_profile`. Existing v2 JSON must still parse unchanged.

- [ ] **Step 2: Verify red** — Run `uv run pytest tests/sol_execbench/core/scoring/test_amd_hardware_models.py tests/sol_execbench/core/scoring/test_amd_sol_v2.py -n 0 -v`. Expected: FAIL because exact v3 profiles do not exist.

- [ ] **Step 3: Implement minimal resolution** — Add v3 profile collections while treating v2 scalar fields as inexact legacy evidence. Remove `VALIDATED_GFX1200_ONLY`; derive validation from calibration provenance plus adapter policy. Add dtype/path/access evidence to work estimates. Require exact compute and memory profiles in bound math; absent/unavailable/unknown profiles emit `unknown_hardware_profile`, degrade confidence, and never fall back to FP32.

- [ ] **Step 4: Verify green and commit** — Run the Step 2 command; expected PASS including legacy fixtures. Commit: `git add src/sol_execbench/core/scoring/amd_hardware_models.py src/sol_execbench/core/scoring/amd_bound_estimate src/sol_execbench/core/scoring/amd_sol tests/sol_execbench/core/scoring/test_amd_hardware_models.py tests/sol_execbench/core/scoring/test_amd_sol_v2.py && git commit -s -m "Resolve bounds from calibrated hardware profiles"`.

### Task 4: Official authority gate and CLI

**Files:**
- Modify: `src/sol_execbench/core/scoring/amd_score/models.py`
- Modify: `src/sol_execbench/core/scoring/amd_score/workload.py`
- Modify: `src/sol_execbench/core/scoring/official_score.py`
- Create: `src/sol_execbench/cli/commands/hardware_model.py`
- Modify: `src/sol_execbench/cli/commands/root.py`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/TESTING.md`
- Modify: `docs/sol_score_gap_and_amd_reuse_report.md`
- Test: `tests/sol_execbench/core/scoring/test_amd_native_score.py`
- Test: `tests/sol_execbench/core/evidence/test_official_score_evidence.py`
- Test: `tests/sol_execbench/cli/commands/test_hardware_model_cli.py`

**Interfaces:** Add `BoundEligibilityEvidence(amd_sol_status, solar_status, hardware_profile_state, hardware_validation_status, model_validation_status, warnings)` to serialized AMD scores. Add `sol-execbench hardware-model calibrate --device 0 --output calibration.json [--architecture GFX] [--require-clock-lock] [--offline] [--no-auto-install]` and `hardware-model build --calibration calibration.json --output calibrated-model.json`.

- [ ] **Step 1: Write failing tests** — A legacy score without `BoundEligibilityEvidence` receives `missing_bound_eligibility` and cannot become official. Exact validated profile evidence passes. CLI writes a rejected artifact before reporting nonzero exit.

- [ ] **Step 2: Verify red** — Run `uv run pytest tests/sol_execbench/core/scoring/test_amd_native_score.py tests/sol_execbench/core/evidence/test_official_score_evidence.py tests/sol_execbench/cli/commands/test_hardware_model_cli.py -n 0 -v`. Expected: FAIL because score evidence and command are absent.

- [ ] **Step 3: Implement propagation and commands** — Derive eligibility from AMD SOL aggregate, SOLAR aggregate, hardware profile, validation states, and warnings. Block `missing_bound_eligibility`, non-scored aggregates, non-supported profiles, non-validated model, and degraded/inexact/unsupported warnings. Add lazy CLI dispatch; write JSON before `ClickException`. Document managed dependencies, external-model selection, and diagnostic versus authoritative results.

- [ ] **Step 4: Verify, run live check, and commit** — Run `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration tests/sol_execbench/core/scoring/test_amd_hardware_models.py tests/sol_execbench/core/scoring/test_amd_sol_v2.py tests/sol_execbench/core/scoring/test_amd_native_score.py tests/sol_execbench/core/evidence/test_official_score_evidence.py tests/sol_execbench/cli/commands/test_hardware_model_cli.py -n 0 -v && uv run --with ruff ruff check .`; expected PASS. Then run the new marker-gated test with `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_live_calibration.py -m requires_rdna4 -n 0 -v`; inspect that profiler gaps are `unknown`. Commit: `git add src tests docs && git commit -s -m "Add calibrated hardware model workflow"`.
