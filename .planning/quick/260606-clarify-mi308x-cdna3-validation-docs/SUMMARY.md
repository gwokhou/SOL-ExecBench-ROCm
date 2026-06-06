# Clarify MI308X CDNA3 Validation Evidence Summary

## Status

Complete.

## Notes

- User clarified that the actual validation machine is MI308X. It shares the
  `gfx942` code target with MI300X, but it has a different hardware
  configuration.
- Documentation must treat MI308X evidence as CDNA3/gfx942 infrastructure
  evidence, not as completed MI300X validation.
- Updated current planning handoff docs, public support/claims docs,
  prerelease scripts, and documentation guardrail tests to preserve this
  boundary.

## Verification

- `uv run pytest tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_public_prerelease_docs.py tests/sol_execbench/test_research_preview_docs.py tests/sol_execbench/test_rocm_support_docs.py tests/sol_execbench/test_public_contract_guardrails.py::test_cdna3_validation_remains_deferred_in_docs tests/sol_execbench/test_prerelease_artifact_bundle.py tests/sol_execbench/test_release_candidate_validation.py tests/sol_execbench/test_prerelease_readiness.py`
- `uv run --with ruff ruff check scripts/build_prerelease_artifact_bundle.py scripts/check_prerelease_readiness.py tests/sol_execbench/test_rocm_support_docs.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_public_prerelease_docs.py tests/sol_execbench/test_research_preview_docs.py tests/sol_execbench/test_prerelease_artifact_bundle.py`
