# Phase 20: Internal Diagnostics and Evidence Model - Plan

**Status:** Ready for execution  
**Created:** 2026-05-22  
**Requirements:** ENG-04, ENG-05, ENG-06

## Objective

Add internal derived evidence/report helpers that combine existing diagnostics
and trace summaries while preserving trace JSONL as canonical benchmark output.

## Scope

In scope:

- Extend pure helpers under `src/sol_execbench/core/`.
- Add derived evidence metadata that labels report objects as non-canonical.
- Add tests for diagnostics, evidence generation, trace non-mutation, and public
  CLI compatibility.

Out of scope:

- No public CLI additions.
- No trace JSONL schema changes.
- No eval-driver replacement.
- No mandatory runtime dependencies.

## Threat Model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-20-01: Derived evidence is mistaken for benchmark output. | High | Evidence model must expose `derived=True` and `canonical_output="trace_jsonl"`. |
| T-20-02: Evidence generation mutates trace objects. | High | Tests compare trace dumps before and after report creation. |
| T-20-03: Diagnostics leak into public CLI or trace schema. | High | Guardrail tests keep CLI/help and trace fields unchanged. |

## Tasks

### Task 20-01: Add Derived Evidence Report Helper

**Requirement:** ENG-05, ENG-06  
**Files:** `src/sol_execbench/core/reporting.py`

1. Add a frozen dataclass for derived evidence reports.
2. Include schema version, derived flag, canonical output label, trace summary,
   and diagnostic rows.
3. Add a pure `build_evidence_report` helper that accepts traces and optional
   diagnostics.

### Task 20-02: Extend Tests

**Requirement:** ENG-04, ENG-05, ENG-06  
**Files:**

- `tests/sol_execbench/test_rocm_diagnostics_reporting.py`
- `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py`

1. Verify evidence reports are derived and canonical output remains trace JSONL.
2. Verify diagnostics are serializable in the evidence report.
3. Verify report generation does not mutate public trace objects.

### Task 20-03: Compatibility Verification

**Requirement:** ENG-06  
**Files:** existing tests only

1. Re-run public contract guardrails.
2. Re-run diagnostics/reporting tests.
3. Run Ruff on changed source/test files.

## Verification

```bash
uv run pytest tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/reporting.py tests/sol_execbench/test_rocm_diagnostics_reporting.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py
```

## Completion Criteria

- Maintainers can inspect internal diagnostics and derived evidence.
- Derived evidence/report output labels itself as non-canonical.
- Existing trace JSONL and CLI contracts remain unchanged.
