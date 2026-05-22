# Phase 20: Internal Diagnostics and Evidence Model - Summary

**Status:** Executed  
**Completed:** 2026-05-22  
**Plan:** `20-PLAN.md`  
**Requirements:** ENG-04, ENG-05, ENG-06

## Delivered

- Added `DerivedEvidenceReport` and `build_evidence_report` in
  `src/sol_execbench/core/reporting.py`.
- Evidence reports combine existing trace summaries and stage diagnostics while
  explicitly labeling themselves as derived.
- Added serialization metadata:
  - `schema_version="sol_execbench.derived_evidence.v1"`
  - `derived=True`
  - `canonical_output="trace_jsonl"`
- Added tests proving evidence generation does not mutate trace objects and
  diagnostics serialize into derived evidence.

## Public Interface Impact

None. This phase did not add public CLI behavior, change Pydantic schemas, add
trace JSONL fields, or alter eval-driver semantics.

## Verification

```bash
uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/reporting.py tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py
```

Result: passed.

## Commits

- `eb5df71` - `feat(20): add derived evidence report`
