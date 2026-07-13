# Phase 34 Context: ROCm Library Category Migration

**Date:** 2026-05-22
**Status:** Complete

## Problem

The schema recognized ROCm library categories (`hipblas`, `miopen`, `ck`,
`rocwmma`), but all were documented as candidates. Public examples prevented
overclaiming, yet no library category had runnable example evidence.

## Relevant Code

- `src/sol_execbench/core/data/solution.py` accepts ROCm native/library
  language values.
- `src/sol_execbench/driver/problem_packager.py` routes native ROCm languages
  through the HIP/C++ build path.
- `src/sol_execbench/driver/templates/build_ext.py` forwards HIP compiler flags
  and linker flags to PyTorch's ROCm extension builder.
- `docs/user/rocm_libraries.md`, `docs/user/solution.md`, and `README.md` define public
  support claims.
- `tests/sol_execbench/test_rocm_library_examples.py` and
  `tests/sol_execbench/test_rocm_library_readiness_docs.py` protect examples
  and support wording.

## Constraints

- Do not claim MIOpen, CK, or rocWMMA support before runnable examples exist.
- Keep former NVIDIA library directories as PyTorch compatibility examples
  unless replaced by real ROCm library implementations.
- Avoid GPU-required tests for CI; verify schema, staging, metadata, and source
  evidence without compiling.
