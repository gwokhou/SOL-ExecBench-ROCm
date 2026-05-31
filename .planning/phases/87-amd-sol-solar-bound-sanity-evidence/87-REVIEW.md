---
phase: 87-amd-sol-solar-bound-sanity-evidence
reviewed: 2026-05-31T11:25:57Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - src/sol_execbench/core/scoring/amd_bound_sanity.py
  - scripts/report_amd_bound_sanity.py
  - tests/sol_execbench/test_amd_bound_sanity.py
  - tests/sol_execbench/test_amd_bound_sanity_script.py
  - tests/sol_execbench/test_public_contract_guardrails.py
  - .planning/phases/87-amd-sol-solar-bound-sanity-evidence/87-CONTEXT.md
  - .planning/phases/87-amd-sol-solar-bound-sanity-evidence/87-01-SUMMARY.md
  - .planning/phases/87-amd-sol-solar-bound-sanity-evidence/87-02-SUMMARY.md
findings:
  critical: 1
  warning: 0
  info: 0
  total: 1
status: issues_found
---

# Phase 87: Code Review Report

**Reviewed:** 2026-05-31T11:25:57Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Reviewed the AMD bound sanity core helper, thin script, Phase 87 tests, public contract guardrails, and phase context/summaries. The implementation stays out of canonical schemas, the primary `sol-execbench` CLI, and AMD-native score eligibility; focused CPU tests pass. However, one checksum normalization bug breaks the sidecar's bounded evidence contract for source refs that use the repository's common `{"value": ...}` checksum shape.

Verification run:

`uv run pytest tests/sol_execbench/test_amd_bound_sanity.py tests/sol_execbench/test_amd_bound_sanity_script.py tests/sol_execbench/test_public_contract_guardrails.py -q`

Result: 47 passed.

## Critical Issues

### CR-01: Source ref checksum objects are serialized as Python dict strings

**Classification:** BLOCKER

**File:** `src/sol_execbench/core/scoring/amd_bound_sanity.py:692`

**Issue:** `_source_from_ref()` checks `value.get("checksum")` with `_optional_str()` before calling `_checksum(value)`. If a trace/source ref uses the repository's common checksum shape, for example `{"checksum": {"value": "abc"}}`, the report emits the string `"{'value': 'abc'}"` instead of `"abc"`. This corrupts bounded source refs/checksums, weakens evidence reproducibility, and defeats the Phase 87 claim that the report carries compact refs/checksums without altering source evidence. Existing tests only cover string `checksum` refs and artifact-specific keys like `report_checksum`, so this contract break is currently untested.

Reproducer:

```python
from sol_execbench.core.scoring.amd_bound_sanity import build_amd_bound_sanity_report

report = build_amd_bound_sanity_report(
    trace_refs=[{"path": "trace.jsonl", "checksum": {"value": "abc"}}],
    created_at="2026-05-31T00:00:00Z",
)
assert report.sources.trace_refs[0].checksum == "abc"
```

Current result is `"{'value': 'abc'}"`.

**Fix:**

Normalize checksums through `_checksum()` before falling back to a raw string value, and add a regression test for nested checksum refs.

```python
return AmdBoundSanitySourceRef(
    path=_optional_str(value.get("path")),
    ref=_optional_str(value.get("ref")),
    schema_version=_optional_str(value.get("schema_version")),
    checksum=_checksum(value),
)
```

If direct string `checksum` is still required, `_checksum()` already handles it.

---

_Reviewed: 2026-05-31T11:25:57Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
