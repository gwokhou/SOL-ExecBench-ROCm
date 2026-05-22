---
phase: 39
status: passed
verified: 2026-05-22
---

# Phase 39 Verification

## Commands

```bash
uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/examples/test_examples.py -k "rocwmma or rocm_library or supported_library or phase_4_example or source_files"
```

Result: 24 passed, 1 skipped.

The skipped test is `test_example[gemm_rocwmma]`. The local machine does not
provide rocWMMA headers under `/opt/rocm/include/rocwmma/rocwmma.hpp`, and it
also lacks the `cuda_runtime_api.h` header needed by PyTorch ROCm native
extension builds. The E2E test is guarded with `requires_rocwmma`,
`requires_rocm_dev`, and `requires_rdna4`.

```bash
uv run --with ruff ruff check tests/conftest.py tests/examples/test_examples.py tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py
```

Result: all checks passed.

## Requirement Coverage

- `WMM-01`: Public rocWMMA example and RDNA 4 E2E registration added; local full
  E2E skipped because rocWMMA/dev headers are incomplete.
- `WMM-02`: Source includes `rocwmma/rocwmma.hpp` and calls fragment load, MMA,
  and store APIs; no PyTorch measured fallback exists.
- `WMM-03`: Tests cover metadata, embedded source consistency, native staging,
  docs/status contract, and RDNA 4 E2E gating.
- `WMM-04`: Documentation now identifies rocWMMA as a supported RDNA 4
  matrix-core GEMM path and explicitly defers CDNA validation.
