---
phase: 40
status: passed
verified: 2026-05-22
---

# Phase 40 Verification

## Commands

```bash
uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/examples/test_examples.py -k "hipblas or miopen or ck or rocwmma or rocm_library or supported_library or phase_4_example or source_files or readme"
```

Result: 38 passed, 3 skipped.

The skipped tests are native library E2E registrations that require complete
ROCm development headers and library headers. This local machine is missing
`cuda_runtime_api.h`, CK headers, and rocWMMA headers, so the tests are
correctly guarded by `requires_rocm_dev`, `requires_ck`, `requires_rocwmma`,
and `requires_rdna4` as applicable.

```bash
uv run --with ruff ruff check tests/conftest.py tests/examples/test_examples.py tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/sol_execbench/test_rocm_diagnostics_reporting.py
```

Result: all checks passed.

## Requirement Coverage

- `COMPAT-01`: Public docs map former NVIDIA categories to supported examples
  or compatibility paths.
- `COMPAT-02`: Former cuDNN, CUTLASS, CuTe DSL, and cuTile paths remain
  compatibility examples unless they contain native ROCm solutions.
- `COMPAT-03`: Public-contract tests enforce supported example coverage for
  MIOpen, CK, and rocWMMA.
- `COMPAT-04`: README and ROCm docs state RDNA 4-only v1.8 validation scope and
  CDNA 3/CDNA 4 deferral.
- `RDNA4-01`: Focused library example suite is registered for RDNA 4 with
  dependency guards; local full E2E is skipped where headers are incomplete.
- `RDNA4-02`: Focused unit/docs tests passed locally.
- `RDNA4-03`: Completion artifacts summarize supported RDNA 4 categories and
  avoid CDNA 3/CDNA 4 validation claims.
