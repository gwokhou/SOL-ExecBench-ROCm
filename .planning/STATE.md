---
gsd_state_version: 1.0
milestone: v1.25
milestone_name: Engineering Prerelease
status: ready_to_plan
last_updated: "2026-06-01"
last_activity: 2026-06-01
progress:
  total_phases: 5
  completed_phases: 2
  total_plans: 2
  completed_plans: 2
  percent: 40
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-01)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.
**Current focus:** v1.25 Engineering Prerelease, ready to plan Phase 116
Claim Boundary Guardrails.

## Current Position

Phase: 116 of 118 (Claim Boundary Guardrails)
Plan: Not planned yet
Status: Ready to plan
Last activity: 2026-06-01 - Phase 115 completed

Progress: [####------] 40%

## Performance Metrics

**Velocity:**
- Total plans completed: 2 in v1.25
- Average duration: n/a
- Total execution time: 0h

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 114. Release-Candidate Validation | 1/1 | Complete | n/a |
| 115. Support Matrix Boundaries | 1/1 | Complete | n/a |
| 116. Claim Boundary Guardrails | 0/TBD | Not started | n/a |
| 117. First-Run User Path | 0/TBD | Not started | n/a |
| 118. Release Candidate Materials | 0/TBD | Not started | n/a |

**Recent Trend:**
- v1.24 shipped Phases 110-113 on 2026-06-01.
- Phase 114 shipped a bounded release-candidate validation wrapper and docs.
- Phase 115 clarified engineering prerelease support boundaries across RDNA 4,
  Docker/container ROCm user-space, MI300X/CDNA3, and unavailable CDNA4.

## Accumulated Context

### Decisions

- v1.25 starts at Phase 114 because v1.24 completed Phase 113.
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

- Plan Phase 116.

### Blockers/Concerns

None.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Paper validation | Full 235-problem paper-scale validation and upstream SOLAR parity | Deferred | v1.25 scope |
| Hardware validation | MI300X/CDNA3 full-suite validation without a complete evidence chain | Deferred | v1.25 scope |
| Hardware validation | CDNA4 validation because suitable hardware is unavailable | Deferred | v1.25 scope |
| Operations | Hosted leaderboard or remote submission service | Deferred | v1.25 scope |
| Security | Hard sandbox or multi-tenant adversarial execution | Deferred | v1.25 scope |
| Dependencies/Docker | Large PyTorch/ROCm relock or Docker privilege redesign | Deferred | v1.25 scope |

## Session Continuity

Last session: 2026-06-01
Stopped at: Phase 115 complete; ready to plan Phase 116.
Resume file: None
