---
phase: 191
status: passed
verified_at: "2026-06-21T09:10:00Z"
verification_type: "goal-backward"
requirements: ["PSUM-01", "PSUM-02", "PSUM-03"]
---

# Phase 191 Verification: Structured Profile Summary Evidence

## Verdict

PASSED. Phase 191 satisfies the structured profile summary goal with CPU-safe, fixture-backed evidence.

## Goal

Expand profile summary into structured profiling evidence with bottleneck hints while preserving diagnostic authority boundaries.

## Success Criteria

1. **Profile summary includes workload/kernel metric records:** VERIFIED
   - `ProfileSummaryContent` now includes `workload_metrics`, `kernel_metrics`, `bottleneck_hints`, and `parse_warnings`.
   - CSV/JSON registered profiler artifacts are parsed with byte and row bounds.
   - Tests cover artifact coverage, dispatch counts, kernel durations, and selected numeric counters.

2. **Bottleneck hints use stable AMD-oriented categories:** VERIFIED
   - Hint categories are schema-restricted to `compute_bound`, `memory_l2_bound`, `lds_bound`, `launch_overhead`, `insufficient_counters`, and `unknown`.
   - Missing counters produce `insufficient_counters`; unmatched counters produce `unknown`.
   - `.rocpd` remains citation-only and does not create speculative hints.

3. **Profile summary cites concrete artifacts:** VERIFIED
   - Phase 190 citation behavior remains intact.
   - Updated fixtures include compact artifact citations and structured evidence.

4. **Governance remains diagnostic-only:** VERIFIED
   - Authority flags remain false except `diagnostic_only`.
   - Claim-upgrade and governance tests continue to reject score, timing, release-gate, cutover, leaderboard, and claim-upgrade authority.

## Requirement Verification

| Requirement | Status | Evidence |
| --- | --- | --- |
| PSUM-01 | Satisfied | Workload/kernel metrics and parse warnings are present in schema, fixtures, and tests. |
| PSUM-02 | Satisfied | Conservative bottleneck hint taxonomy is schema-restricted and tested. |
| PSUM-03 | Satisfied | Authority guardrail tests and docs preserve diagnostic-only semantics. |

## Automated Checks

- PASS: `uv run pytest tests/sol_execbench/test_profile_summary.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_profile_summary_fixtures.py`
  - Result: 51 passed.
- PASS: `uv run --with ruff ruff check src/sol_execbench/core/bench/profile_summary.py src/sol_execbench/cli/main.py tests/sol_execbench/test_profile_summary.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_profile_summary_fixtures.py`
  - Result: All checks passed.
- PASS: implementation spot-check with `rg` confirmed structured metric fields, hint vocabulary, fixture coverage, and citation-only `.rocpd` behavior.

## Decision Coverage

All 9 trackable CONTEXT.md decisions are honored by shipped artifacts.

## Residual Risk

Real GPU profiler validation was not run in this CPU-safe execution. The parser is intentionally limited to bounded CSV/JSON artifacts; `.rocpd` database parsing and fine-grained occupancy/VGPR/SGPR/cache/bandwidth diagnostics remain deferred.
