---
status: complete
---

# Summary

Updated current CDNA3/gfx942 validation documentation after the cloud
full-validation run and nested timeout classification fix.

## Changes

- Reframed CDNA3 status from purely deferred to validation infrastructure
  evidence with known blockers.
- Recorded the corrected dataset interpretation: 220 complete passing problem
  traces, 15 expected Quant NVFP4/MXFP4 CDNA3 skips, and 6 timeout shards across
  4 problems.
- Preserved the claim boundary that full CDNA3/MI300X hardware validation is not
  complete until timeout, clock-lock, timing, score, FP8, and low-precision
  evidence boundaries are resolved.
- Updated public, internal, prerelease, release-candidate, solution, ROCm, and
  handoff docs.
- Updated document guardrail tests for the new boundary wording.

## Verification

- `uv run pytest tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_public_prerelease_docs.py tests/sol_execbench/test_research_preview_docs.py tests/sol_execbench/test_rocm_support_docs.py tests/sol_execbench/test_public_contract_guardrails.py::test_cdna3_validation_remains_deferred_in_docs tests/sol_execbench/test_prerelease_artifact_bundle.py tests/sol_execbench/test_release_candidate_validation.py tests/sol_execbench/test_prerelease_readiness.py`
- `uv run --with ruff ruff check tests/sol_execbench/test_rocm_support_docs.py tests/sol_execbench/test_public_contract_guardrails.py`
