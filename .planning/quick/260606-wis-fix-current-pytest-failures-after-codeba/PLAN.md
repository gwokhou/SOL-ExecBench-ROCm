---
status: in_progress
created_at: "2026-06-06T15:24:57.138Z"
quick_id: 260606-wis
slug: fix-current-pytest-failures-after-codeba
---

# Fix Current Pytest Failures After Codebase Remap

## Objective

Restore the current test suite to green after the codebase remap changed
documentation wording and evidence sidecar naming expectations.

## Scope

- Update the stale dataset run-closure fixture to use the current namespaced
  sidecar stem helper.
- Restore public/codebase documentation guardrail phrases expected by tests.
- Re-run the failing tests, then the full suite if the focused checks pass.

## Verification

- `uv run pytest tests/sol_execbench/test_dataset_run_closure.py::test_derived_evidence_for_workload_combines_present_refs_and_missing_gaps tests/sol_execbench/test_research_release_docs.py::test_v1_21_docs_keep_debt_reduction_separate_from_external_claims tests/sol_execbench/test_rocm_test_suite_audit.py::test_hardware_markers_do_not_create_mi300x_or_cdna4_validation_shortcuts tests/sol_execbench/test_rocm_library_readiness_docs.py::test_readme_links_library_readiness_and_names_supported_libraries -q`
- `uv run pytest tests/ -q`
