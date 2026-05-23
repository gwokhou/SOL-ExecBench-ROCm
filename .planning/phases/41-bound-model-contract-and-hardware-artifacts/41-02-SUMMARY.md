# 41-02 Summary

## Wave
02

## Scope Completed
- Replaced in-`amd_sol` hard-coded hardware constants with packaged-loader compatibility facade.
- Updated artifact/schema serialization to expose split validation statuses.
- Updated score warnings to treat either hardware status or model status as unvalidated.
- Updated `tests/sol_execbench/test_amd_sol_bounds.py` and `tests/sol_execbench/test_amd_native_score.py` for CDNA3 external-model handling and v2 split-status assertions.

## Acceptance
- `default_amd_hardware_models()` now resolves from packaged `gfx1200.json` and returns only `gfx1200`.
- Derived artifacts remain compatible and no longer expose legacy `validation_status`.

## Verification
- `uv run pytest tests/sol_execbench/test_amd_sol_bounds.py -x`
- `uv run pytest tests/sol_execbench/test_amd_native_score.py -x`
