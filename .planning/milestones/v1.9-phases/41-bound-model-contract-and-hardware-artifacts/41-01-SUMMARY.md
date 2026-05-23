# 41-01 Summary

## Wave
01

## Scope Completed
- Added packaged AMD hardware model resource under `src/sol_execbench/data/amd_hardware_models/gfx1200.json`.
- Added package modules for `sol_execbench.data` and `sol_execbench.data.amd_hardware_models`.
- Implemented strict v2 loader/parser in `src/sol_execbench/core/scoring/amd_hardware_models.py`.
- Added unit tests in `tests/sol_execbench/test_amd_hardware_models.py` for packed defaults, external model loading, unknown-field rejection, deprecated `validation_status`, schema validation, and architecture-policy checks.

## Acceptance
- v2 artifact contract uses `hardware_validation_status` + `model_validation_status`, no `validation_status`.
- Unknown fields and invalid values are rejected explicitly.
- `load_packaged_amd_hardware_model("gfx1200")` and `default_amd_hardware_models()` are wired through strict parser path.

## Verification
- `uv run pytest tests/sol_execbench/test_amd_hardware_models.py -x`
