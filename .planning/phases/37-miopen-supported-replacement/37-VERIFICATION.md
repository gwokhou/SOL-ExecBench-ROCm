---
phase: 37
status: passed
verified: 2026-05-22
---

# Phase 37 Verification

## Commands

```bash
uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/examples/test_examples.py -k "miopen or rocm_library or supported_library or phase_4_example or source_files"
```

Result: 20 passed, 1 skipped.

The skipped test is `test_example[softmax_miopen]`. The local machine exposes a
ROCm/RDNA 4 path but does not provide `cuda_runtime_api.h`, which PyTorch's
native extension headers require for compiling ROCm C++ extensions. The test is
guarded with `requires_rocm_dev` so full RDNA 4 E2E remains active on machines
with complete ROCm development headers.

```bash
uv run --with ruff ruff check tests/conftest.py tests/examples/test_examples.py tests/sol_execbench/test_rocm_library_examples.py
```

Result: all checks passed.

## Requirement Coverage

- `MIOPEN-01`: Public MIOpen example and RDNA 4 E2E registration added; local
  full E2E compile skipped because ROCm dev headers are incomplete.
- `MIOPEN-02`: Source includes `miopen/miopen.h` and calls
  `miopenSoftmaxForward_V2`; no PyTorch measured fallback exists.
- `MIOPEN-03`: Tests cover metadata, embedded source consistency, native
  staging, docs/status contract, and RDNA 4 E2E gating.
- `MIOPEN-04`: Documentation now identifies MIOpen as the supported softmax
  replacement and describes descriptor constraints.
