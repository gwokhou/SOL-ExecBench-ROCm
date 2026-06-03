# Phase 127 Summary: CDNA3 Hardware-Gated Test Surface

**Completed:** 2026-06-04
**Status:** Complete

## Delivered

- Added `tests/sol_execbench/test_cdna3_hardware_marker.py` with a concrete
  `@pytest.mark.requires_cdna3` live hardware-gated test.
- Added CPU-safe tests for CDNA3 marker registration, `gfx94*` architecture
  detection, RDNA4 skip behavior, missing-ROCm skip behavior, and allowed
  `gfx94*` marker selection.
- Strengthened `tests/sol_execbench/test_rocm_test_suite_audit.py` with an AST
  audit that fails if no direct `requires_cdna3` test exists.
- Added schema-level CDNA3 offload-arch metadata coverage for `gfx940`,
  `gfx941`, and `gfx942` without creating a validation claim.

## Verification

- `uv run pytest tests/sol_execbench/test_cdna3_hardware_marker.py tests/sol_execbench/test_rocm_test_suite_audit.py tests/sol_execbench/core/data/test_solution.py`
  - `93 passed, 1 skipped`
- `uv run pytest tests/sol_execbench/test_cdna3_hardware_marker.py -m requires_cdna3 -n 0`
  - `1 skipped, 10 deselected`
- `uv run --with ruff ruff check tests/sol_execbench/test_cdna3_hardware_marker.py tests/sol_execbench/test_rocm_test_suite_audit.py tests/sol_execbench/core/data/test_solution.py`
  - passed

## Deferred

- Real CDNA3/MI300X execution remains deferred until a `gfx94*` ROCm host is
  available.
- The new live test is marker/readiness coverage, not full CDNA3 hardware
  validation evidence.
