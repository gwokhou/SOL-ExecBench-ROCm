---
phase: 38
status: passed
verified: 2026-05-22
---

# Phase 38 Verification

## Commands

```bash
uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/examples/test_examples.py -k "ck or rocm_library or supported_library or phase_4_example or source_files"
```

Result: 25 passed, 1 skipped.

The skipped test is `test_example[gemm_ck]`. The local machine does not provide
Composable Kernel headers under `/opt/rocm/include/ck/ck.hpp`, and it also lacks
the `cuda_runtime_api.h` header needed by PyTorch ROCm native extension builds.
The E2E test is guarded with `requires_ck`, `requires_rocm_dev`, and
`requires_rdna4`.

```bash
uv run --with ruff ruff check tests/conftest.py tests/examples/test_examples.py tests/sol_execbench/test_rocm_library_examples.py
```

Result: all checks passed.

## Requirement Coverage

- `CK-01`: Public CK example and RDNA 4 E2E registration added; local full E2E
  skipped because CK/dev headers are incomplete.
- `CK-02`: Source includes CK headers and uses CK index/tiling conventions; no
  PyTorch measured fallback exists.
- `CK-03`: Tests cover metadata, embedded source consistency, native staging,
  docs/status contract, and RDNA 4 E2E gating.
- `CK-04`: Documentation now identifies CK as a supported path for the small
  GEMM example and explains scope limits.
