---
quick_id: 260606-clarify-cdna3-mi300x-hierarchy
slug: clarify-cdna3-mi300x-hierarchy
status: complete
completed: 2026-06-06
---

# Clarify CDNA3 And MI300X Hierarchy Summary

## Status

Complete.

## Notes

- User clarified that CDNA 3 and MI300X are not peer concepts. CDNA 3 is the
  architecture; MI300X and MI308X are peer GPU products under that architecture.

## Changes

- Replaced current `CDNA3/MI300X` and `MI300X-as-CDNA3` wording with hierarchy
  wording.
- Public docs now describe MI300X and MI308X as sibling GPU products under the
  CDNA3 architecture family that share the `gfx942` code path.
- Updated prerelease scripts and tests so generated/readiness wording preserves
  the same hierarchy.

## Verification

- `uv run pytest tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_public_prerelease_docs.py tests/sol_execbench/test_research_preview_docs.py tests/sol_execbench/test_rocm_support_docs.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_prerelease_artifact_bundle.py tests/sol_execbench/test_release_candidate_validation.py tests/sol_execbench/test_prerelease_readiness.py tests/sol_execbench/test_rocm_test_suite_audit.py`
- `uv run --with ruff ruff check scripts/build_prerelease_artifact_bundle.py scripts/check_prerelease_readiness.py scripts/release_candidate_validation.py tests/sol_execbench/test_rocm_support_docs.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_public_prerelease_docs.py tests/sol_execbench/test_research_preview_docs.py tests/sol_execbench/test_prerelease_artifact_bundle.py tests/sol_execbench/test_release_candidate_validation.py tests/sol_execbench/test_rocm_test_suite_audit.py`
