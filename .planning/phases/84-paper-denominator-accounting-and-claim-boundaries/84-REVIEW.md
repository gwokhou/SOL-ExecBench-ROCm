---
phase: 84-paper-denominator-accounting-and-claim-boundaries
reviewed: 2026-05-31T08:42:42Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - scripts/report_paper_denominator.py
  - src/sol_execbench/core/dataset/__init__.py
  - src/sol_execbench/core/dataset/paper_denominator.py
  - tests/sol_execbench/test_paper_denominator_report.py
  - tests/sol_execbench/test_paper_denominator_script.py
  - tests/sol_execbench/test_public_contract_guardrails.py
findings:
  critical: 2
  warning: 2
  info: 0
  total: 4
status: findings_found
---

# Phase 84: Code Review Report

**Reviewed:** 2026-05-31T08:42:42Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** findings_found

## Summary

Reviewed the Phase 84 paper denominator sidecar, script wrapper, dataset exports, and CPU-safe tests. The implementation is sidecar-only and the focused suite passes, but the report builder can under-report missing evidence and emit internally inconsistent denominator totals. Those are contract-level correctness failures for a report whose purpose is denominator accounting and claim-boundary discipline.

Verification run during review:

`UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_paper_denominator_report.py tests/sol_execbench/test_paper_denominator_script.py tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_paper_denominator_fields_remain_sidecar_only tests/sol_execbench/test_public_contract_guardrails.py::test_primary_cli_does_not_expose_v1_19_paper_denominator_options -q`

Result: `9 passed in 1.36s`

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: BLOCKER - Missing Evidence Is Not Accounted When Optional Evidence Sources Are Absent

**File:** `src/sol_execbench/core/dataset/paper_denominator.py:546`

**Issue:** The builder only creates evidence-gap buckets from pre-existing `execution_closure.records[*].evidence_gaps` and unsupported AMD score rows. If `amd_score_report` is omitted, or if no `--amd-sol-artifact` / `--solar-artifact` is supplied, lines 607-619 still emit empty source refs but no `evidence_missing` state, reason bucket, evidence gap, or next-evidence hint. That violates the Phase 84 contract that missing AMD score, AMD SOL, SOLAR, timing, or trace evidence must become `evidence_missing` or `deferred`, never silently disappear.

**Impact:** A researcher can generate a report with no AMD score/SOL/SOLAR sources and no corresponding gap. The report then understates missing evidence and can appear more complete than the bounded source refs justify.

**Fix:** After reading closure records and optional source refs, derive required evidence completeness from `trace_ref`, `evidence_refs`, `amd_score_report`, `amd_sol_artifacts`, and `solar_artifacts`. Add missing buckets for absent report-level sources and per-workload missing refs.

```python
required_refs = {
    "timing": "timing_evidence",
    "amd_score": "amd_score",
    "amd_sol": "amd_sol_bound",
    "solar_derivation": "solar_derivation",
}
for record in execution_closure.get("records", []):
    if str(record.get("closure_status")) in {"filtered", "not_attempted"}:
        continue
    refs = record.get("evidence_refs", {})
    for evidence, ref_key in required_refs.items():
        if not refs.get(ref_key):
            reason_code = f"{evidence}_evidence_missing"
            example_ref = _record_ref(record)
            _add_reason(
                reason_groups,
                reason_code=reason_code,
                state="evidence_missing",
                example_ref=example_ref,
            )
            _add_evidence_gap(
                evidence_groups,
                reason_code=reason_code,
                example_ref=example_ref,
            )
            # also add evidence_missing to workload/category/problem rollups once
```

Also add tests where `amd_score_report=None`, `amd_sol_artifacts=[]`, and `solar_artifacts=[]` produce explicit missing-evidence buckets.

### CR-02: BLOCKER - Suite Totals Diverge From Emitted Problem And Workload Records

**File:** `src/sol_execbench/core/dataset/paper_denominator.py:460`

**Issue:** Suite/category problem totals are seeded from inventory category denominators at lines 460-464, readiness workloads increment only readiness workload counts at lines 466-505, and closure records add new `PaperDenominatorProblem` / `PaperDenominatorWorkload` entries at lines 507-562 without incrementing corresponding workload totals. The current fixture report demonstrates the bug: `suite.problems == 3` and `suite.workloads == 4`, while the emitted `problems` and `workloads` arrays each contain 8 records.

**Impact:** The report is internally contradictory. Markdown "Suite Counts" can disagree with the machine-readable record arrays and state buckets, making denominator accounting unreliable for researcher review.

**Fix:** Normalize around one denominator source of truth. Seed problem/workload records from `inventory["problems"]` and their nested `workloads`, then merge readiness and closure attributes onto those records by stable `(problem_id, row_index, workload_uuid)` keys. Derive suite/category/problem/workload counts from the normalized records or from inventory denominators, but do not mix both without reconciliation.

```python
# Pseudocode shape
for problem in inventory.get("problems", []):
    problem_rollup = _problem_rollup(...)
    problem_rollup.problems = 1
    for workload in problem.get("workloads", []):
        key = _workload_key(problem["problem_id"], workload)
        workloads[key] = PaperDenominatorWorkload(...)
        problem_rollup.workloads += 1
        _category_rollup(categories, category).workloads += 1

# Later readiness/closure loops should update states only, not create extra
# denominator records unless explicitly classified as out-of-inventory evidence.
```

Add assertions that `suite.problems == len(report.problems)` and `suite.workloads == len(report.workloads)` for fixture inputs where every emitted record is in-scope.

## Warnings

### WR-01: WARNING - Tests Mask The Denominator Mismatch And Missing Optional Evidence Cases

**File:** `tests/sol_execbench/test_paper_denominator_report.py:301`

**Issue:** The main aggregation test asserts fixed suite totals and only checks that selected problem/workload IDs are a superset at lines 301-328. It does not assert exact problem/workload arrays or that suite totals reconcile with those arrays. The fixture also always supplies AMD score, AMD SOL, and SOLAR inputs at lines 276-288, so missing-source behavior is untested.

**Impact:** The current tests pass while the report emits contradictory denominator totals and silently accepts absent optional evidence sources.

**Fix:** Tighten the contract tests:

```python
payload = build_fixture_report().model_dump(mode="json")
assert payload["suite"]["problems"] == len(payload["problems"])
assert payload["suite"]["workloads"] == len(payload["workloads"])

missing = build_paper_denominator_report(
    inventory=inventory_fixture(),
    readiness=readiness_fixture(),
    execution_closure=execution_closure_fixture(),
    amd_score_report=None,
    amd_sol_artifacts=[],
    solar_artifacts=[],
    created_at=CREATED_AT,
).model_dump(mode="json")
assert "amd_score_evidence_missing" in {g["reason_code"] for g in missing["evidence_gaps"]}
assert "amd_sol_evidence_missing" in {g["reason_code"] for g in missing["evidence_gaps"]}
assert "solar_derivation_missing" in {g["reason_code"] for g in missing["evidence_gaps"]}
```

### WR-02: WARNING - Markdown Renderer Does Not Escape Untrusted Table Cells

**File:** `src/sol_execbench/core/dataset/paper_denominator.py:659`

**Issue:** Source paths/refs, reason codes, example refs, and next-evidence text are interpolated directly into Markdown table cells at lines 659-669 and 724-744. These values originate from local sidecar dictionaries and CLI paths. A value containing `|`, newlines, or Markdown control text can break table structure or inject misleading rows into the researcher-facing summary.

**Impact:** The JSON remains structured, but the Markdown summary can be corrupted or misleading. This matters because Phase 84 explicitly makes Markdown a deterministic researcher review artifact with visible claim-boundary wording.

**Fix:** Sanitize Markdown table cells before interpolation and add tests with paths/reasons containing pipes and newlines.

```python
def _md_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")

# Then use _md_cell(...) for every dynamic table cell.
```

---

_Reviewed: 2026-05-31T08:42:42Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
