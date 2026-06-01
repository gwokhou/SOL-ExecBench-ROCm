---
gsd_state_version: 1.0
milestone: v1.26
milestone_name: Public Prerelease and Research Preview
status: executing
last_updated: "2026-06-01T16:52:00.000Z"
last_activity: 2026-06-01
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 3
  completed_plans: 3
  percent: 75
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-01)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.
**Current focus:** v1.26 Public Prerelease and Research Preview, ready to plan
Phase 122 Public Publishing Materials.

## Current Position

Phase: 122 of 122 (Public Publishing Materials)
Plan: Not planned yet
Status: Ready to plan
Last activity: 2026-06-01 — Phase 121 completed with research preview evidence package

Progress: [███████---] 75%

## Recent Trend

- v1.23 shipped Phases 106-109 on 2026-06-01.
- v1.24 shipped Phases 110-113 on 2026-06-01.
- v1.25 shipped Phases 114-118 on 2026-06-01.

## Accumulated Context

### Decisions

- v1.25 is an engineering prerelease / release-candidate milestone, not a
  paper-scale validation or hosted-service milestone.

- MI300X is the CDNA3 hardware target; MI300X and CDNA3 are not separate
  validation targets.

- CDNA4 validation is unavailable because suitable hardware is not currently
  accessible.

- Full 235-problem paper-scale validation, upstream SOLAR parity, hosted
  leaderboard, hard sandboxing, large dependency relocking, and Docker
  privilege redesign remain deferred unless explicitly reopened.

### Pending Todos

- Plan Phase 122.

### Blockers/Concerns

None.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Paper validation | Full 235-problem paper-scale validation and upstream SOLAR parity | Deferred | v1.26 scope |
| Hardware validation | Full MI300X validation on the CDNA3 `gfx942` target without a complete evidence chain | Deferred | v1.26 scope |
| Hardware validation | CDNA4 validation because suitable hardware is unavailable | Deferred | v1.26 scope |
| Operations | Hosted leaderboard or remote submission service | Deferred | v1.26 scope |
| Security | Hard sandbox or multi-tenant adversarial execution | Deferred | v1.26 scope |
| Release authority | Stable benchmark authority release | Deferred | v1.26 scope |

## Session Continuity

Last session: 2026-06-01
Stopped at: v1.26 initialized; ready to plan Phase 119.
Resume file: None
