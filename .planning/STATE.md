---
gsd_state_version: 1.0
milestone: v1.35
milestone_name: Script Parallelism and Safety Hardening
status: milestone_complete
stopped_at: Milestone complete (Phase 180 was final phase, gaps closed)
last_updated: 2026-06-11T05:30:00Z
last_activity: 2026-06-11
progress:
  total_phases: 6
  completed_phases: 6
  total_plans: 7
  completed_plans: 7
  percent: 100
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-11)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.
**Current focus:** Planning next milestone

## Current Position

Phase: 180
Plan: Complete
Status: Milestone complete
Last activity: 2026-06-11

Progress: [██████████] 100%

## Recent Trend

- v1.35 shipped on 2026-06-11. 6 phases, 7 plans, 23 requirements satisfied.
  Added PID locks, timing isolation, CPU-parallel staging, parallel dispatch,
  evaluation stability reason codes, GPU device isolation, strict-isolation
  mode, and rocprofv3 overhead calibration.

- v1.34 shipped on 2026-06-09. Phases 170-174 reduced RDNA4 readiness_blocked
  from 114 to 59 over a stable 235-problem denominator.

## Quick Tasks Completed

| Date | Task | Status | Notes |
|------|------|--------|-------|
| 2026-06-13 | rdna4-validation-100-percent | complete | Added explicit `--target-status` selection to the RDNA4 profiler batch so the latest v1.35 report's 73 `ready_missing_profiler_timing` problems can be targeted for follow-up profiling; strict full profiler-backed timing remains blocked by 41 readiness-blocked and 5 reference OOM-blocked denominator rows. |
| 2026-06-12 | rdna4-v135-measurement-rerun | complete | Rebuilt RDNA4 v1.35 rerun closure, derived evidence, profiler timing batch, coverage, denominator, consistency, claim, trust, and bundle reports under `out/rdna4-v135-rerun-20260611/`; fixed derived/profiler OOM, profiler ENOSPC, and consistency evidence-gap drift misclassification. |

## Accumulated Context

### Decisions

- fcntl.flock selected over PID-in-file locking for kernel-managed auto-release.
- ThreadPoolExecutor chosen over ProcessPoolExecutor due to torch fork-safety.
- CPU-parallel staging + GPU-serial profiling architecturally enforced.
- Overhead calibration uses inner subprocess under rocprofv3.

### Pending Todos

None.

### Blockers/Concerns

None.

## Deferred Items

Items acknowledged and deferred at prior milestone closes:

| Category | Item | Status |
|----------|------|--------|
| Paper validation | Full 235-problem paper-scale validation and upstream SOLAR parity | Deferred |
| Hardware validation | Full MI300X validation on the CDNA3 `gfx942` target | Deferred |
| Hardware validation | CDNA4 validation because suitable hardware is unavailable | Deferred |
| Operations | Hosted leaderboard or remote submission service | Deferred |
| Security | Hard sandbox or multi-tenant adversarial execution | Deferred |
| Release authority | Stable benchmark authority release | Deferred |
| Dataset redistribution | Publishing or hosting NVIDIA/SOL-ExecBench original or derivative dataset content | Deferred |

## Session Continuity

Last session: 2026-06-11T05:30:00Z
Stopped at: v1.35 milestone complete, planning next milestone
Resume file: None

## Operator Next Steps

- Start next milestone: `/gsd:new-milestone`
