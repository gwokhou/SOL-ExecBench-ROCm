---
phase: 97
status: complete
completed: 2026-06-01
---

# Phase 97 Summary: SOLAR Derivation And Static Evidence Modularization

## Completed

- Added `sol_execbench.core.scoring.solar_derivation_status` for SOLAR status
  mapping, status count ordering, default source boundary, deterministic unique
  ordering, and derivation warning construction.
- Updated `solar_derivation.py` to delegate those pure status and boundary
  decisions while preserving existing sidecar models and parser behavior.
- Added `sol_execbench.core.bench.static_kernel_status` for parser-independent
  static extractor aggregate status and reason selection.
- Updated `static_kernel_evidence.py` to delegate extractor aggregate status and
  reason decisions while preserving sidecar status/reason outputs.
- Added focused helper tests in the existing SOLAR and static evidence suites.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -q`  
  Result: 82 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py -q`  
  Result: 22 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_solar_derivation_family_modeling.py -q`  
  Result: 30 passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/scoring/solar_derivation.py src/sol_execbench/core/scoring/solar_derivation_status.py src/sol_execbench/core/bench/static_kernel_evidence.py src/sol_execbench/core/bench/static_kernel_status.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_static_kernel_evidence.py`  
  Result: passed.

## Notes

This phase preserves all SOLAR and static evidence sidecar schemas. It extracts
status, source-boundary, warning, and aggregate outcome decisions first; deeper
parser or extractor execution decomposition remains possible without changing
public payloads.
