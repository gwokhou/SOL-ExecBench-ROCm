---
phase: 51-sidecar-coverage-and-score-guards
reviewed: 2026-05-23T09:24:44Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - src/sol_execbench/core/scoring/solar_derivation.py
  - src/sol_execbench/core/scoring/amd_score.py
  - tests/sol_execbench/test_solar_derivation_evidence.py
  - tests/sol_execbench/test_amd_native_score.py
  - tests/sol_execbench/test_public_contract_guardrails.py
  - tests/sol_execbench/test_amd_sol_v2.py
findings:
  critical: 1
  warning: 1
  info: 0
  total: 2
status: issues_found
---

# Phase 51: Code Review Report

**Reviewed:** 2026-05-23T09:24:44Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Reviewed the Phase 51 sidecar coverage, aggregate status, parser, AMD-native score guard, and public-boundary test changes. Planning summaries under `.planning/phases/51-sidecar-coverage-and-score-guards/` were used as context only.

The score guard path mostly implements the requested unscored/degraded precedence, but the sidecar aggregate contract is internally inconsistent for degraded evidence, and the Phase 51 parser accepts semantically corrupted coverage/aggregate fields instead of rejecting them.

## Critical Issues

### CR-01: BLOCKER - Degraded aggregate status says scores are ineligible even though degraded scores must remain numeric when inputs are complete

**File:** `src/sol_execbench/core/scoring/solar_derivation.py:2245`

**Issue:** `_aggregate_status_for_groups()` emits `status="degraded"` with `score_eligible=False` for any degraded-only aggregate. That conflicts with the Phase 51 requirement that degraded evidence preserves warnings while still allowing a numeric AMD-native score when measured latency, baseline latency, and SOL bound inputs are complete. The score code follows that requirement by suppressing only explicit `unscored` aggregates and computing a score for degraded aggregates when numeric inputs are complete (`src/sol_execbench/core/scoring/amd_score.py:187` and `src/sol_execbench/core/scoring/amd_score.py:192`). The sidecar contract therefore tells downstream consumers that the same degraded evidence is not score eligible while the scorer reports `supported=True`.

**Evidence:** `tests/sol_execbench/test_amd_native_score.py:230` covers numeric scoring for a manually constructed degraded aggregate, but there is no test that passes a derived degraded `SolarDerivationEvidence` through `score_amd_native_workload()` or asserts `evidence.to_dict()["aggregate_status"]["score_eligible"] is True` for degraded-only evidence.

**Fix:**

```python
if any(group.status == "degraded" for group in groups):
    return SolarAggregateStatus(
        status="degraded",
        score_eligible=True,
        reason="one or more semantic groups have incomplete evidence",
        group_ids=group_ids,
        node_ids=node_ids,
        warnings=aggregate_warnings,
    )
```

Add a regression test that builds a degraded-only `SolarDerivationEvidence`, asserts `aggregate_status.status == "degraded"` and `score_eligible is True`, then passes that evidence object to `score_amd_native_workload()` and verifies the numeric score plus degraded warnings.

## Warnings

### WR-01: WARNING - Phase 51 parser accepts corrupted coverage and aggregate fields instead of rejecting semantic mismatches

**File:** `src/sol_execbench/core/scoring/solar_derivation.py:613`

**Issue:** When Phase 51 fields are present, `solar_derivation_from_dict()` calls `_coverage_summary_from_dict()` and `_aggregate_status_from_dict()` only for structural validation, discards both parsed objects, and returns a `SolarDerivationEvidence` that recomputes those fields from `groups` and `warnings` on serialization (`src/sol_execbench/core/scoring/solar_derivation.py:625`). This accepts payloads whose `coverage_summary.family_counts`, `status_counts`, `missing_patterns`, or `aggregate_status.status` contradict the groups. That violates the parser guard intent for malformed nested Phase 51 fields and can hide stale or tampered sidecars while making round-trip output silently differ from input.

**Evidence:** The malformed parser tests at `tests/sol_execbench/test_solar_derivation_evidence.py:856` mutate types, missing fields, or unknown fields, but they do not mutate a validly typed nested value to be semantically wrong. A payload can set `aggregate_status.status` to `"scored"` or zero out `coverage_summary.status_counts` for an unscored group and still parse successfully because lines 613-623 never compare parsed fields to the recomputed expected sidecar.

**Fix:** For Phase 51 payloads, parse the provided summary/status and compare their `to_dict()` output with `_coverage_for_groups(groups).to_dict()` and `_aggregate_status_for_groups(groups, warnings).to_dict()`. Raise `ValueError` on mismatch. Keep the existing legacy path (`raw_keys == legacy_keys`) as the only path that recomputes missing Phase 51 fields without requiring an input match.

---

_Reviewed: 2026-05-23T09:24:44Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
