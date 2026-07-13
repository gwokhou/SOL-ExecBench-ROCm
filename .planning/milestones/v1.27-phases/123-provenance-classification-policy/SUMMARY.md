# Phase 123 Summary: Provenance Classification Policy

**Status:** Complete
**Completed:** 2026-06-02

## Delivered

- Added `provenance.toml` as a machine-readable provenance policy and NVIDIA
  notice classification artifact.
- Added `docs/user/provenance.md` explaining classification classes, header policy,
  paper citation boundaries, non-endorsement wording, and history policy.
- Added `tests/sol_execbench/test_provenance_policy.py` to verify the manifest
  accounts for all current active NVIDIA SPDX header files.

## Requirements Completed

- PROV-01
- PROV-02
- PROV-03

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_provenance_policy.py -q`
  - 4 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check tests/sol_execbench/test_provenance_policy.py`
  - passed

## Follow-Up

Phase 124 should consume `provenance.toml` to apply header cleanup. It should
review cleanup candidates before replacing NVIDIA-only attribution with project
attribution.
