---
gsd_state_version: 1.0
milestone: v1.28
milestone_name: CDNA3 Test and Documentation Readiness
status: planning
last_updated: "2026-06-04T00:00:00+08:00"
last_activity: 2026-06-04 — Phase 127 completed
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 4
  completed_plans: 1
  percent: 25
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-01)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.
**Current focus:** v1.28 CDNA3 Test and Documentation Readiness is planned.

## Current Position

Phase: 128 MI300X Evidence Contract and Validation Handoff
Plan: —
Status: Ready for phase discussion or planning
Last activity: 2026-06-04 — Phase 127 completed

## Recent Trend

- v1.23 shipped Phases 106-109 on 2026-06-01.
- v1.24 shipped Phases 110-113 on 2026-06-01.
- v1.25 shipped Phases 114-118 on 2026-06-01.
- v1.26 shipped Phases 119-122 on 2026-06-02.
- Quick task 260602-jjy fixed GitHub Actions Ty/CPU-safe CI failures on
  2026-06-02.

- Quick task 260602-mbt added a pre-commit-managed git pre-push Ty check hook
  on 2026-06-02.

- Quick task 260602-miz made `pre-commit install` enable pre-commit,
  commit-msg, and pre-push hooks by default on 2026-06-02.

- Quick task 260602-mmo added `pre-commit` to the dev dependency group on
  2026-06-02.

- Quick task 260602-mqi fixed stale CI, Docker, pre-commit, dependency marker,
  and documentation configuration on 2026-06-02.

- Quick task 260602-mqr tightened hook locking, Ruff excludes, Docker runtime
  dependency groups, Linux x86_64 ROCm markers, and configuration docs on
  2026-06-02.

- Quick task 260602-msd removed unnecessary README prose line wrapping while
  preserving command formatting on 2026-06-02.

- Quick task 260603-ffd resolved remaining GSD health warnings by indexing
  archived v1.27 phases and moving validation handoff files into the milestone
  archive on 2026-06-03.
- v1.28 CDNA3 Test and Documentation Readiness started on 2026-06-04 with
  Phase 127-130 scoped for CDNA3 hardware-gated tests, MI300X evidence
  contracts, deferred-validation guardrails, and public documentation closure.
- Phase 127 completed on 2026-06-04 with concrete `requires_cdna3` hardware
  marker tests, CPU-safe skip behavior checks, AST marker-use audit coverage,
  and CDNA3 metadata tests.

## Accumulated Context

### Decisions

- v1.25 is an engineering prerelease / release-candidate milestone, not a
  paper-scale validation or hosted-service milestone.

- MI300X is the CDNA3 hardware target; MI300X and CDNA3 are not separate
  validation targets.

- v1.28 should add real CDNA3 test readiness and documentation without claiming
  actual CDNA3 hardware validation on the current machine.

- CDNA4 validation is unavailable because suitable hardware is not currently
  accessible.

- Full 235-problem paper-scale validation, upstream SOLAR parity, hosted
  leaderboard, hard sandboxing, large dependency relocking, and Docker
  privilege redesign remain deferred unless explicitly reopened.

### Pending Todos

None.

### Blockers/Concerns

None.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Paper validation | Full 235-problem paper-scale validation and upstream SOLAR parity | Deferred | v1.26 scope |
| Hardware validation | Full MI300X validation on the CDNA3 `gfx942` target without a complete evidence chain | Deferred | v1.26 scope |
| Hardware validation | Actual CDNA3/MI300X full-suite execution during v1.28 on the current machine | Deferred | v1.28 scope |
| Hardware validation | CDNA4 validation because suitable hardware is unavailable | Deferred | v1.26 scope |
| Operations | Hosted leaderboard or remote submission service | Deferred | v1.26 scope |
| Security | Hard sandbox or multi-tenant adversarial execution | Deferred | v1.26 scope |
| Release authority | Stable benchmark authority release | Deferred | v1.26 scope |

## Session Continuity

Last session: 2026-06-04
Stopped at: Phase 127 complete; ready for Phase 128.
Resume file: None

## Operator Next Steps

- Run `/gsd-discuss-phase 128` to gather implementation context.
- Or run `/gsd-plan-phase 128` to plan Phase 128 directly.
