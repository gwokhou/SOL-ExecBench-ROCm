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

