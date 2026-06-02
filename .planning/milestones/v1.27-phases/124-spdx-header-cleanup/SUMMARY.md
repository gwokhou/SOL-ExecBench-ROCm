# Phase 124 Summary: SPDX Header Cleanup

**Status:** Complete
**Completed:** 2026-06-02

## Delivered

- Updated all files in `provenance.toml` `nvidia_notice.allowed` to preserve
  NVIDIA attribution and add project attribution.
- Updated all files in `provenance.toml` `nvidia_notice.cleanup_candidates` to
  replace NVIDIA-only attribution with project attribution.
- Preserved Apache-2.0 SPDX license identifiers and script shebang lines.
- Updated provenance policy tests to assert the cleaned target state.
- Updated the ROCm migration residue audit so legitimate PyTorch ROCm CUDA
  compatibility fields and provenance tests are explicitly classified.

## Requirements Completed

- COPY-01
- COPY-02
- COPY-03
- COPY-04

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_provenance_policy.py tests/sol_execbench/test_rocm_migration_residue_audit.py -q`
  - 7 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check tests/sol_execbench/test_provenance_policy.py tests/sol_execbench/test_rocm_migration_residue_audit.py`
  - passed

## Follow-Up

Phase 125 should align public docs and release wording with the cleaned
provenance policy.
