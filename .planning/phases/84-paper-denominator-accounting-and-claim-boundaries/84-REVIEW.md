---
phase: 84-paper-denominator-accounting-and-claim-boundaries
reviewed: 2026-05-31T08:48:23Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - src/sol_execbench/core/dataset/paper_denominator.py
  - tests/sol_execbench/test_paper_denominator_report.py
findings:
  critical: 1
  warning: 0
  info: 0
  total: 1
status: findings_found
---

# Phase 84: Code Review Report

**Reviewed:** 2026-05-31T08:48:23Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** findings_found

## Summary

Re-reviewed Phase 84 after commit `6de979f`, focusing on the prior CR-01, CR-02, WR-01, and WR-02 findings plus regressions from the fix. CR-01, WR-01, and WR-02 are resolved: absent optional sources now produce evidence-missing buckets, per-workload missing refs are accounted for closure records, the focused tests cover those cases, and Markdown table cells are escaped.

One blocker remains from CR-02. The new totals reconciliation makes suite/category/problem/workload totals match the currently emitted arrays, but it still does not seed denominator records from `inventory["problems"]` and nested `workloads`. Inventory-only denominator entries disappear from both emitted arrays and suite totals.

Verification run:

`UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_paper_denominator_report.py tests/sol_execbench/test_paper_denominator_script.py tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_paper_denominator_fields_remain_sidecar_only tests/sol_execbench/test_public_contract_guardrails.py::test_primary_cli_does_not_expose_v1_19_paper_denominator_options -q`

Result: `11 passed in 1.36s`

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: BLOCKER - Inventory Denominator Problems And Workloads Can Still Disappear From The Report

**File:** `src/sol_execbench/core/dataset/paper_denominator.py:501`

**Issue:** The fix still never seeds `problems` or `workloads` from `inventory["problems"]` and their nested `workloads`. Lines 501-505 briefly read category denominator problem counts, but lines 669-675 reset every category's `problems` and `workloads` to counts derived only from records created by the readiness and execution-closure sidecars. If inventory contains a parsed problem/workload that has no readiness or closure row, the report emits `suite.problems == 0`, `suite.workloads == 0`, `problems == []`, and `workloads == []` for that denominator entry.

This keeps the core CR-02 correctness failure partially open. The current tests now assert `suite.problems == len(report.problems)` and `suite.workloads == len(report.workloads)`, but that only proves internal consistency after dropping inventory-only denominator entries. It does not prove that the paper denominator report covers the inventory denominator.

Concrete repro:

```python
report = build_paper_denominator_report(
    inventory={
        "schema_version": "sol_execbench.dataset_inventory.v1",
        "categories": [
            {"name": "L1", "denominators": {"parsed_problems": 1, "parsed_workloads": 2}},
        ],
        "problems": [
            {
                "category": "L1",
                "problem_id": "L1/only_inventory",
                "problem_path": "L1/only_inventory",
                "workload_count": 2,
                "workloads": [
                    {"uuid": "w0", "row_index": 0},
                    {"uuid": "w1", "row_index": 1},
                ],
            },
        ],
    },
    readiness={"workloads": []},
    execution_closure={"records": []},
    created_at="2026-05-31T00:00:00Z",
)
assert report.suite.problems == 1
assert report.suite.workloads == 2
```

Observed output from the current implementation: `{'problems': 0, 'workloads': 0, ...}`, `report.problems == []`, and `report.workloads == []`.

**Fix:** Normalize from inventory first, then merge readiness and closure attributes onto those records by stable keys. Category and suite problem/workload totals should be derived from the normalized inventory-backed records, not from whichever readiness/closure rows happen to exist.

```python
for problem in inventory.get("problems", []):
    category = str(problem.get("category", "unknown"))
    problem_id = str(problem.get("problem_id") or problem.get("problem_path") or "unknown")
    problem_path = str(problem.get("problem_path")) if problem.get("problem_path") else None
    _problem_rollup(
        problems,
        category=category,
        problem_id=problem_id,
        problem_path=problem_path,
    )
    for workload in problem.get("workloads", []):
        key = (
            problem_id,
            workload.get("row_index"),
            str(workload.get("uuid")) if workload.get("uuid") else None,
        )
        workloads.setdefault(
            key,
            PaperDenominatorWorkload(
                category=category,
                problem_id=problem_id,
                problem_path=problem_path,
                workload_uuid=str(workload.get("uuid")) if workload.get("uuid") else None,
                row_index=workload.get("row_index"),
            ),
        )
```

Then add a regression test where inventory contains a parsed problem/workload not present in readiness or execution closure, and assert it remains in `report.problems`, `report.workloads`, category counts, and suite counts.

---

_Reviewed: 2026-05-31T08:48:23Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
