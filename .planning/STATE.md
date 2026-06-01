---
gsd_state_version: 1.0
milestone: v1.25
milestone_name: Engineering Prerelease
status: ready_for_audit
last_updated: "2026-06-01"
last_activity: 2026-06-01
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 5
  completed_plans: 5
  percent: 100
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-01)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.
**Current focus:** v1.25 Engineering Prerelease, all phases complete and ready
for milestone audit.

## Current Position

Phase: 118 of 118 (Release Candidate Materials)
Plan: 118-01 complete
Status: Ready for milestone audit
Last activity: 2026-06-01 - Phase 118 completed

Progress: [##########] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 5 in v1.25
- Average duration: n/a
- Total execution time: 0h

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 114. Release-Candidate Validation | 1/1 | Complete | n/a |
| 115. Support Matrix Boundaries | 1/1 | Complete | n/a |
| 116. Claim Boundary Guardrails | 1/1 | Complete | n/a |
| 117. First-Run User Path | 1/1 | Complete | n/a |
| 118. Release Candidate Materials | 1/1 | Complete | n/a |

**Recent Trend:**
- v1.24 shipped Phases 110-113 on 2026-06-01.
- Phase 114 shipped a bounded release-candidate validation wrapper and docs.
- Phase 115 clarified engineering prerelease support boundaries across RDNA 4,
  Docker/container ROCm user-space, MI300X/CDNA3, and unavailable CDNA4.
- Phase 116 added v1.25 release notes and claim-boundary guardrails for
  canonical, diagnostic-only, provisional, deferred, and unavailable evidence.
- Phase 117 clarified the first-run user path, trace interpretation, no-trace
  diagnostics, known limitations, and PyTorch ROCm compatibility wording.
- Phase 118 added release-candidate checklist, release note navigation, README
  entry points, and release-material guardrails.

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

- Run milestone audit.

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
Stopped at: All v1.25 phases complete; ready for milestone audit.
Resume file: None
