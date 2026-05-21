---
phase: 05-rocm-test-suite-and-hardware-validation
plan: 01
subsystem: pytest-markers
tags:
  - rocm
  - pytest
  - hardware
key-files:
  - tests/conftest.py
  - pyproject.toml
metrics:
  tests: "11 passed in focused marker audit suite"
---

# Plan 05-01 Summary

## Changes

- Replaced NVIDIA SM detection with ROCm GPU detection through PyTorch's HIP
  enabled `torch.cuda` compatibility API.
- Added `requires_rocm`, `requires_rdna4`, and `requires_cdna3` marker
  semantics.
- Preserved `requires_cutile` as a legacy NVIDIA-only marker that is skipped in
  the ROCm-only port.
- Updated pytest marker descriptions from C++/CUDA and Blackwell wording to
  HIP/C++ and AMD gfx architecture wording.

## Verification

- `uv run --no-sync pytest tests/sol_execbench/test_rocm_test_suite_audit.py tests/sol_execbench/test_rocm_library_examples.py tests/examples/test_examples.py -k consistency` -> 11 passed.
- `uv run --no-sync ruff check ...` on changed Python files -> passed.

## Deviations

- PyTorch still exposes ROCm devices through `torch.cuda`; the code checks
  `torch.version.hip` before treating that compatibility API as ROCm.

## Self-Check: PASSED

Marker behavior now distinguishes unavailable ROCm, unsupported AMD gfx
architecture, RDNA 4, and CDNA 3.

