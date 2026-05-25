# Quick Audit Fix Summary: Remaining Suite Reds

## Delivered

- Corrected MIOpen library readiness to require standard `libMIOpen`.
- Kept `PYTORCH_ROCM_ARCH` template behavior while satisfying the import
  guardrail.
- Restored documentation and planning compatibility phrases required by
  public-contract guardrails.
- Classified current legitimate CUDA/NVIDIA residue contexts instead of
  deleting boundary evidence.

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -n 0 tests/docker/dependencies/test_rocm_libraries.py tests/sol_execbench/driver/test_build_ext.py tests/sol_execbench/test_hip_execbench_practice_map.py tests/sol_execbench/test_non_cdna_validation_closure.py tests/sol_execbench/test_original_parity_docs.py tests/sol_execbench/test_rocm_test_suite_audit.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/sol_execbench/test_rocm_migration_residue_audit.py -q
```

Result: `50 passed in 0.72s`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check src/sol_execbench/core/diagnostics.py src/sol_execbench/driver/templates/build_ext.py tests/sol_execbench/test_rocm_migration_residue_audit.py
```

Result: `All checks passed!`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -q
```

Result: `906 passed, 58 skipped in 140.81s`.
