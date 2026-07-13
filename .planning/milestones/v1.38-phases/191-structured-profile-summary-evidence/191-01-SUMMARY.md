---
phase: 191
plan: "191-01"
subsystem: "profile_summary.sidecar.v1 structured evidence"
tags: ["profile-summary", "rocprofv3", "diagnostic-evidence", "bottleneck-hints"]
requires: ["PSUM-01", "PSUM-02", "PSUM-03"]
provides:
  - "structured workload and kernel profile metrics"
  - "conservative diagnostic bottleneck hints"
  - "HIP-facing profile summary fixtures with structured evidence"
affects:
  - "src/sol_execbench/core/bench/profile_summary.py"
  - "tests/sol_execbench/test_profile_summary.py"
  - "tests/sol_execbench/test_cli_environment_snapshot.py"
  - "tests/sol_execbench/test_profile_summary_fixtures.py"
  - "tests/sol_execbench/fixtures/profile_summary/"
  - "docs/user/profile_summary_sidecar.md"
  - "docs/user/agent_feedback_sidecar.md"
tech-stack:
  added: []
  patterns:
    - "bounded CSV/JSON artifact parsing"
    - "diagnostic-only sidecar authority guardrails"
    - "closed taxonomy bottleneck hints"
key-files:
  created: []
  modified:
    - "src/sol_execbench/core/bench/profile_summary.py"
    - "tests/sol_execbench/test_profile_summary.py"
    - "tests/sol_execbench/test_cli_environment_snapshot.py"
    - "tests/sol_execbench/test_profile_summary_fixtures.py"
    - "tests/sol_execbench/fixtures/profile_summary/valid.profile-summary.json"
    - "tests/sol_execbench/fixtures/profile_summary/partial.profile-summary.json"
    - "tests/sol_execbench/fixtures/profile_summary/unavailable.profile-summary.json"
    - "tests/sol_execbench/fixtures/profile_summary/stale.profile-summary.json"
    - "docs/user/profile_summary_sidecar.md"
    - "docs/user/agent_feedback_sidecar.md"
key-decisions:
  - "CSV and JSON artifacts are parsed with byte and row bounds; rocpd/database artifacts remain citation-only."
  - "Bottleneck hints use the conservative closed taxonomy and degrade to insufficient_counters or unknown when evidence is incomplete."
  - "Profile summary remains diagnostic-only and cannot satisfy score, release-gate, cutover, or claim-upgrade authority."
requirements-completed: ["PSUM-01", "PSUM-02", "PSUM-03"]
duration: "15 min"
completed: "2026-06-21"
---

# Phase 191 Plan 01: Structured Profile Metrics and Diagnostic Bottleneck Hints Summary

Expanded `profile_summary.sidecar.v1` with structured workload/kernel metrics, bounded CSV/JSON profiler artifact parsing, conservative bottleneck hints, and updated HIP-facing fixtures/docs while preserving diagnostic-only authority.

## Execution

- Started: 2026-06-21T08:53:41Z
- Completed: 2026-06-21T09:08:48Z
- Tasks completed: 6/6
- Files modified: 10

## Commits

| Task | Commit | Description |
| --- | --- | --- |
| T01 | `76b8ae3` | Added failing tests defining structured profile evidence and hint behavior. |
| T02-T04 | `f940461` | Added strict structured evidence models, bounded CSV/JSON parsing, and conservative hint derivation. |
| T05 | `1ebe2fc` | Updated CLI sidecar fixture behavior and profile-summary fixtures. |
| T06 | `f61dd2d` | Documented structured profile summary evidence and HIP downgrade semantics. |

## Verification

- PASS: `uv run pytest tests/sol_execbench/test_profile_summary.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_profile_summary_fixtures.py`
  - Result: 51 passed.
- PASS: `uv run --with ruff ruff check src/sol_execbench/core/bench/profile_summary.py src/sol_execbench/cli/main.py tests/sol_execbench/test_profile_summary.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_profile_summary_fixtures.py`
  - Result: All checks passed.

Real GPU profiler validation was not run in this CPU-safe phase execution. The parser is covered with fixture-backed CSV/JSON artifacts and intentionally treats `.rocpd` database artifacts as citation-only.

## Deviations From Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Self-Check: PASSED

Phase 191 now satisfies the must-haves: CSV/JSON artifacts are parsed with bounded logic, `.rocpd` remains citation-only, workload/kernel metrics include coverage, dispatch, duration, counters, parse status and warnings, bottleneck hints use the conservative taxonomy, fixtures/docs/tests are updated together, and authority flags remain diagnostic-only.

## Next

Phase 191 is ready for verification and then Phase 192 planning.
