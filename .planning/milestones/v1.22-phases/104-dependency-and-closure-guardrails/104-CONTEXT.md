# Phase 104: Dependency And Closure Guardrails - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

## Phase Boundary

Maintainers can catch policy, provenance, and marker regressions before they
create misleading ROCm validation signals.

This phase covers requirements GUARD-01 through GUARD-03:

- Keep ROCm wheel, Docker target, and dependency-matrix policy consistency
  guarded when target metadata changes.
- Cover execution-closure provenance for sidecar refs, stale provenance, and
  manifest/cache provenance behavior.
- Keep hardware-marker skip behavior explicit so CPU-safe green runs cannot be
  mistaken for RDNA4, CDNA3, timing, MI300X, or CDNA4 validation.

## Current Code Context

- Dependency policy coverage lives in
  `tests/sol_execbench/test_dependency_matrix_policy.py` and checks
  `docker/rocm-targets.json`, `pyproject.toml`, `uv.lock`, and matrix payloads.
- Execution closure contract coverage lives in
  `tests/sol_execbench/test_execution_closure_contract.py` and already checks
  status vocabularies, provenance mismatch reason codes, checksums, and unknown
  field rejection.
- Runtime dataset closure coverage in
  `tests/sol_execbench/test_run_dataset_execution_closure.py` covers broader
  CLI/helper behavior.
- Marker behavior is centralized in `tests/conftest.py` and audited by
  `tests/sol_execbench/test_rocm_test_suite_audit.py`.
- `tests/conftest.py` dynamically registers timing and ROCm headers markers,
  skips `timing_serial` unless selected, skips architecture-specific tests based
  on detected gfx architecture, and always skips legacy `requires_cutile`.

## Implementation Direction

Add focused CPU-safe tests rather than changing runtime behavior. The likely
implementation surface is limited to tests unless the guardrail reveals a real
logic gap.

## Verification Targets

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_execution_closure_contract.py tests/sol_execbench/test_rocm_test_suite_audit.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_execution_closure_contract.py tests/sol_execbench/test_rocm_test_suite_audit.py`

