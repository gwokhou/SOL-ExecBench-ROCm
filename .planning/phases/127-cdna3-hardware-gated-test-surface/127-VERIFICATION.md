---
status: passed
phase: 127
phase_name: CDNA3 Hardware-Gated Test Surface
verified_at: 2026-06-04
---

# Phase 127 Verification

## Result

Passed.

## Must-Haves

- CDNA3 hardware-gated test exists and uses `@pytest.mark.requires_cdna3`.
  - Verified by `tests/sol_execbench/test_cdna3_hardware_marker.py`.
- Non-CDNA3 and missing-ROCm hosts skip with explicit reasons.
  - Verified by CPU-safe monkeypatched collection tests.
- CPU-safe audit fails if no concrete direct `requires_cdna3` test exists.
  - Verified by AST audit in `tests/sol_execbench/test_rocm_test_suite_audit.py`.
- CDNA3 metadata/build support remains separate from hardware validation.
  - Verified by schema/offload metadata tests and requirements wording checks.

## Commands

```bash
uv run pytest tests/sol_execbench/test_cdna3_hardware_marker.py tests/sol_execbench/test_rocm_test_suite_audit.py tests/sol_execbench/core/data/test_solution.py
uv run pytest tests/sol_execbench/test_cdna3_hardware_marker.py -m requires_cdna3 -n 0
uv run --with ruff ruff check tests/sol_execbench/test_cdna3_hardware_marker.py tests/sol_execbench/test_rocm_test_suite_audit.py tests/sol_execbench/core/data/test_solution.py
```

## Notes

The `requires_cdna3` live test skipped on the current host, which is expected
because this machine does not expose real `gfx94*` ROCm hardware.
