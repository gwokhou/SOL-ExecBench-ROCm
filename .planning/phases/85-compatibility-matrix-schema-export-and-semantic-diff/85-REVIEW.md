---
phase: 85-compatibility-matrix-schema-export-and-semantic-diff
reviewed: 2026-05-31T09:40:06Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - scripts/diff_matrix_reports.py
  - scripts/export_matrix_schema.py
  - src/sol_execbench/core/compatibility.py
  - src/sol_execbench/core/matrix_diff.py
  - tests/sol_execbench/test_matrix_claim_guardrails.py
  - tests/sol_execbench/test_matrix_schema_export.py
  - tests/sol_execbench/test_matrix_semantic_diff.py
findings:
  critical: 2
  warning: 1
  info: 0
  total: 3
status: findings_found
---

# Phase 85: Code Review Report

**Reviewed:** 2026-05-31T09:40:06Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** findings_found

## Summary

Reviewed the Phase 85 Matrix schema export and semantic diff implementation across core helpers, scripts, and CPU-safe tests. The schema export surface is narrow and deterministic, but the diff contract has two blocking correctness issues around diagnostic authority enforcement and Docker image metadata severity classification. The Markdown renderer also allows Matrix string payloads to break table structure.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Diff result model accepts authoritative claim flags

**File:** `src/sol_execbench/core/matrix_diff.py:115`

**Severity:** BLOCKER

**Issue:** `MatrixReportDiff` documents that diff output is diagnostic-only, but the authority fields are plain `bool` fields. A caller or downstream validator can construct or accept a `MatrixReportDiff` with `diagnostic_compatibility_evidence=False`, `score_authority=True`, `paper_parity_authority=True`, `leaderboard_authority=True`, or `native_host_validation_authority=True`, and `to_dict()` will emit those authoritative claims. This violates the Phase 85 claim-boundary requirement that diff outputs cannot upgrade Docker/container evidence into native-host, score, paper-parity, or leaderboard authority.

**Impact:** Any consumer that treats `MatrixReportDiff.model_validate(...)` as the strict diff artifact contract can accept a tampered or incorrectly constructed authoritative diff payload. The current tests only assert the helper-produced defaults, not that the model rejects invalid authority flags.

**Suggested fix:**

```python
from typing import Literal

class MatrixReportDiff(BaseModelWithDocstrings):
    diagnostic_compatibility_evidence: Literal[True] = True
    score_authority: Literal[False] = False
    paper_parity_authority: Literal[False] = False
    leaderboard_authority: Literal[False] = False
    native_host_validation_authority: Literal[False] = False
```

Add a regression test that `MatrixReportDiff.model_validate(...)` rejects true authority fields and `diagnostic_compatibility_evidence=False`.

### CR-02: Requested Docker image metadata drift is misclassified as generic semantic change

**File:** `src/sol_execbench/core/matrix_diff.py:305`

**Severity:** BLOCKER

**Issue:** `_image_dependency_drift()` only checks `observed.container`, `observed.python_dependency`, and `observed.dependency_policy`. Requested Docker image metadata lives under `target.docker_image_repository` and `target.docker_image_tag`, so a Target-only image change lands in the broad `target` semantic group and receives only `semantic_change` severity. Phase 85 explicitly requires severity-ranked image/dependency drift for Docker image metadata, not just observed container metadata.

**Impact:** A Matrix report can change its requested Docker image repository/tag without the JSON or Markdown severity categories surfacing `image_dependency_drift`. Downstream reviewers and gates that look for image/dependency drift will miss a required semantic category.

**Suggested fix:**

```python
def _image_dependency_drift(changes: dict[str, dict[str, Any]]) -> bool:
    if any(
        group in changes
        for group in (
            "observed.container",
            "observed.python_dependency",
            "observed.dependency_policy",
        )
    ):
        return True

    target_change = changes.get("target")
    if target_change is None:
        return False
    old_target = target_change["old"] or {}
    new_target = target_change["new"] or {}
    return (
        old_target.get("docker_image_repository") != new_target.get("docker_image_repository")
        or old_target.get("docker_image_tag") != new_target.get("docker_image_tag")
    )
```

Add a regression test where only `target.docker_image_tag` changes and assert `image_dependency_drift` appears in `severity_categories`.

## Warnings

### WR-01: Markdown table renderer does not escape user-controlled cell values

**File:** `src/sol_execbench/core/matrix_diff.py:500`

**Severity:** WARNING

**Issue:** `_markdown_value()` returns raw `json.dumps(...)` output, and the renderer inserts it directly into Markdown table cells at lines 531 and 555. Matrix fields such as `target_id`, artifact paths, artifact URIs, descriptions, image tags, and reason strings are user/report-controlled strings. If any value contains `|`, the generated Markdown table structure is split into extra columns, making old/new values ambiguous or misleading.

**Impact:** The machine-readable JSON remains valid, but the human-readable summary can become malformed or misleading for valid Matrix payloads. This weakens the deterministic human review surface Phase 85 added.

**Suggested fix:** Escape Markdown table delimiters before inserting cell values, and test with an artifact path or target string containing `|`.

```python
def _markdown_value(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True).replace("\\", "\\\\").replace("|", "\\|")
```

---

_Reviewed: 2026-05-31T09:40:06Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
