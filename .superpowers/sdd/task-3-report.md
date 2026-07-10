# Task 3 report — Hardware model v3 and exact SOL profile resolution

## Delivered

- Added strict v3 hardware-model profiles (`HardwareProfile`) and exact compute/
  memory lookup APIs while retaining v2 JSON parsing and serialization.
- Bound estimates now record normalized dtype, compute operation/path, and memory
  access evidence. Bounds use only matching measured profiles; missing,
  unavailable, or unknown evidence produces `unknown_hardware_profile` and a
  degraded result without a scalar/FP32 fallback.
- Removed the architecture-specific `gfx1200` validation restriction. Calibration
  validation now records provenance from adapter policy, clock lock state, and
  measured declared profiles.

## Verification

```text
uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_hardware_models.py src/sol_execbench/core/scoring/amd_bound_estimate src/sol_execbench/core/scoring/amd_sol src/sol_execbench/core/scoring/hardware_calibration/builder.py tests/sol_execbench/core/scoring/test_amd_hardware_models.py tests/sol_execbench/core/scoring/test_amd_sol_v2.py
# All checks passed!

uv run pytest tests/sol_execbench/core/scoring/test_amd_hardware_models.py tests/sol_execbench/core/scoring/test_amd_sol_v2.py tests/sol_execbench/core/scoring/hardware_calibration/test_builder.py -n 0 -v
# 22 passed
```

## Review follow-up

- Memory profiles now resolve by the complete access, input dtype, output dtype,
  and path key. A BF16 workload therefore cannot consume an FP32 stream profile.
- Bound confidence now applies the general confidence ordering to every resolved
  profile; an `unsupported` profile makes the operation unsupported and the
  aggregate unscored.

Focused review verification: `17 passed` for the Task 3 scoring tests.
