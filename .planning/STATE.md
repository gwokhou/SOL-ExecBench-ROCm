---
gsd_state_version: 1.0
milestone: v1.19
milestone_name: Research Credibility Without New Hardware
status: Awaiting next milestone
stopped_at: Completed 88-02-PLAN.md
last_updated: "2026-05-31T13:16:51.787Z"
last_activity: 2026-05-31 — Milestone v1.19 completed and archived
progress:
  total_phases: 6
  completed_phases: 6
  total_plans: 13
  completed_plans: 13
  percent: 100
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-31)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.
**Current focus:** Planning next milestone

## Current Position

Phase: Milestone v1.19 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-05-31 — Milestone v1.19 completed and archived

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: n/a
- Total execution time: 0h

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 83. Closure Contracts And Provenance Foundation | 0/TBD | Not started | n/a |
| 84. Paper Denominator Accounting And Claim Boundaries | 0/TBD | Not started | n/a |
| 85. Compatibility Matrix Schema Export And Semantic Diff | 0/TBD | Not started | n/a |
| 86. Dataset Runner Hardening Integration | 0/TBD | Not started | n/a |
| 87. AMD SOL/SOLAR Bound Sanity Evidence | 0/TBD | Not started | n/a |
| 88. Documentation, Examples, And Guardrail Tests | 0/TBD | Not started | n/a |

**Recent Trend:**

- Last milestone: v1.18 shipped Phases 78-82 on 2026-05-28.
- Trend: v1.19 starts with six standard-granularity phases focused on sidecar/reporting credibility without new hardware validation.

| Phase 83 P01 | 403s | 2 tasks | 2 files |
| Phase 83 P02 | 403s | 3 tasks | 4 files |
| Phase 84 P01 | 352 | 3 tasks | 2 files |
| Phase 84 P02 | 263 | 3 tasks | 4 files |
| Phase 85 P01 | 215 | 3 tasks | 3 files |
| Phase 85 P02 | 491 | 3 tasks | 4 files |
| Phase 86 P01 | 358 | 3 tasks | 4 files |
| Phase 86 P02 | 185 | 3 tasks | 4 files |
| Phase 87 P02 | 265 | 3 tasks | 3 files |
| Phase 87 P01 | 1069 | 3 tasks | 2 files |
| Phase 88 P01 | 960 | 3 tasks | 5 files |
| Phase 88 P02 | 780 | 2 tasks | 11 files |

## Accumulated Context

### Decisions

- v1.19 starts at Phase 83 because v1.18 completed Phases 78-82.
- v1.19 derives phases only from v1.19 requirements: closure, denominator, Matrix tooling, runner hardening, AMD bound sanity, and docs/tests.
- v1.19 must not expand CDNA3, MI300X, CDNA4, or native-host hardware validation.
- v1.19 evidence remains sidecar/reporting infrastructure and must not change canonical Trace, Definition, Workload, Solution, correctness, timing, scoring, or evaluator semantics.
- Denominator, Matrix, Docker, and AMD SOL/SOLAR reports must keep paper parity, score authority, leaderboard authority, native-host validation, and new-hardware validation claims false unless future evidence upgrades them.
- [Phase 83]: scripts/run_dataset.py delegates execution closure status, totals, record validation, report construction, and writing to core helpers without adding resume/reuse enforcement.
- [Phase 83]: Execution closure remains a sidecar-only sol_execbench.execution_closure.v1 contract.
- [Phase 84]: [Phase 84 Plan 01]: Paper denominator accounting remains a sidecar-only report and does not change canonical benchmark schemas.
- [Phase 84]: [Phase 84 Plan 01]: Missing evidence is accounted as evidence_missing/deferred and never upgraded into validation, score, leaderboard, paper, SOLAR, native-host, or new-hardware authority.
- [Phase 84]: [Phase 84 Plan 02]: Optional AMD SOL and SOLAR artifacts remain bounded source refs/checksums, not embedded payloads.
- [Phase 84]: [Phase 84 Plan 02]: Paper denominator reports are exposed through a script and dataset helper exports, not primary sol-execbench CLI options.
- [Phase 85]: [Phase 85 Plan 01]: Schema export remains script-side diagnostic tooling, not a primary sol-execbench CLI option.
- [Phase 85]: [Phase 85 Plan 01]: Matrix schema exports are limited to MatrixEntry and RocmCompatibilityMatrixReport.
- [Phase 85]: [Phase 85 Plan 02]: Diff JSON and Markdown remain diagnostic-only with score, paper-parity, leaderboard, and native-host validation authority false.
- [Phase 85]: [Phase 85 Plan 02]: Matrix report diffs match entries by target_id plus validation_scope.
- [Phase 86]: Raw CLI stdout/stderr remains in per-problem log files while execution_closure.json stores relative cli_log_ref plus concise notes. — Keeps closure sidecars bounded and avoids embedding raw logs or absolute temp paths.
- [Phase 86]: Existing passing traces authorize skipped_existing_pass only when output/execution_closure.json contains matching provenance. — Prevents stale or tampered sidecars from authorizing clean reuse.
- [Phase 87]: [Phase 87 Plan 02]: AMD bound sanity generation is exposed only as a research script, not as a primary sol-execbench CLI option or package entry point.
- [Phase 87]: [Phase 87 Plan 02]: The report script loads only explicitly supplied JSON paths and delegates normalization, checksum, status, and rendering behavior to core helpers.
- [Phase 87]: [Phase 87 Plan 01]: amd_bound_sanity.v1 remains a scoring sidecar/reporting artifact and does not modify score eligibility or canonical schemas.
- [Phase 87]: [Phase 87 Plan 01]: Claim boundaries are literal false fields plus visible Markdown wording; provisional RDNA 4 risk is a risk flag, not validation.
- [Phase 88]: [Phase 88]: v1.19 evidence guidance is centralized in docs/v1_19_evidence_guide.md and linked from CLAIMS, TESTING, and RESEARCHER-GUIDE.
- [Phase 88]: [Phase 88]: v1.19 fixtures remain demo-only with synthetic checksums, relative refs, bounded log refs, diagnostic-only wording, and false authority fields.

### Pending Todos

None found.

### Blockers/Concerns

None.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Hardware validation | CDNA 3, MI300X, CDNA 4, and native-host ROCm validation expansion | Deferred | v1.19 scope |
| Paper parity | Full 235-problem real-hardware validation and upstream SOLAR parity | Deferred | v1.19 scope |
| Public service | Hosted leaderboard or remote submission service | Deferred | v1.19 scope |
| Dependencies | PyTorch/ROCm dependency relocking or Docker privilege expansion | Deferred | v1.19 scope |

## Quick Tasks Completed

| Date | Quick Task | Result |
|------|------------|--------|
| 2026-05-31 | 260531-u2s add requires_rocm coverage for CLI and dataset runner GPU paths | Added 3 ROCm E2E regressions; `requires_rocm` now reports 17 passed. |
| 2026-05-31 | 260531-uki add remaining requires_rocm e2e coverage | Added HIP/C++ CLI, static evidence, and run_dataset reuse/rerun E2E regressions; `requires_rocm` now reports 19 passed. |

## Session Continuity

Last session: 2026-05-31T12:17:09.178Z
Stopped at: Completed 88-02-PLAN.md
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
