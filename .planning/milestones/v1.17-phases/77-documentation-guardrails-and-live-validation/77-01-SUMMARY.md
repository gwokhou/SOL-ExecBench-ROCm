# Plan 77-01 Summary: Documentation, Guardrails, And Live Validation

**Completed:** 2026-05-26  
**Status:** Complete  
**Commit:** `a18a75a #77 - Document static evidence boundaries`

## What Changed

- Added `docs/user/static_kernel_evidence.md` as the public guide for
  `--static-evidence none|auto`, sidecar paths, schema identity, status
  vocabulary, and diagnostic-only claim boundaries.
- Updated `docs/user/CLAIMS.md` with allowed Static Kernel Evidence wording,
  required evidence, forbidden authority claims, and upgrade rules.
- Updated `docs/user/RESEARCHER-GUIDE.md` and `docs/user/rocm_toolchain_routing.md` so
  researcher and routing docs describe the v1.17 static evidence surface.
- Added `docs/internal/v1_17_static_kernel_evidence_validation.md` with bounded
  RDNA 4 validation results.
- Added documentation guardrail tests for usage, deferred scope, and validation
  artifact boundaries.

## Live Validation

- Environment detected `hipcc`, `rocminfo`, `rocm-smi`, and RDNA 4 `gfx1200`.
- The HIP/C++ RMSNorm example compiled and produced a collected static evidence
  sidecar with 9 artifacts and 6 tool runs.
- The benchmark command exited nonzero because all 14 workloads returned
  `RUNTIME_ERROR` with `hidden_states must be a HIP tensor`.
- The validation artifact therefore claims only static evidence collection on
  this RDNA 4 environment, not benchmark correctness, timing, performance,
  score validity, paper parity, or leaderboard readiness.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_public_contract_guardrails.py -q`
  - Result: 40 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_toolchain_routing.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_public_contract_guardrails.py -q`
  - Result: 84 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check tests/sol_execbench/test_research_release_docs.py`
  - Result: All checks passed

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| SKE-DOCS-01 | Complete | Public guide documents opt-in usage, sidecar paths, schema, status vocabulary, and diagnostic-only interpretation. |
| SKE-DOCS-02 | Complete | Claims and researcher guide document allowed language, required evidence, and forbidden authority claims. |
| SKE-DOCS-03 | Complete | Internal validation artifact records RDNA 4 static evidence collection and benchmark correctness failure boundary. |
| SKE-DOCS-04 | Complete | Documentation tests guard usage, claim boundaries, deferred scopes, and validation artifact wording. |
| SKE-DOCS-05 | Complete | Deferred CDNA 3, CDNA 4, Triton cache, RGA-rich parsing, and paper-scale coverage are explicit. |

## Deferred

- CDNA 3 and CDNA 4 live validation require hardware-specific archived runs.
- Triton ROCm cache capture is not implemented.
- RGA-rich VGPR/SGPR/LDS/scratch/occupancy parsing remains deferred.
- Paper-scale static coverage remains out of scope for v1.17.
