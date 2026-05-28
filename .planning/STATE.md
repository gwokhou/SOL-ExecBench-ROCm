---
gsd_state_version: 1.0
milestone: v1.18
milestone_name: ROCm Version Matrix via Docker
status: planning
last_updated: "2026-05-28"
last_activity: 2026-05-28
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-28)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.
**Current focus:** Phase 78 - Matrix Contract And Claim Guardrails

## Current Position

Phase: 78 of 82 (Matrix Contract And Claim Guardrails)
Plan: TBD
Status: Ready to plan
Last activity: 2026-05-28 - Created v1.18 roadmap from 30 requirements

Progress: [----------] 0%

## Performance Metrics

**Velocity:**
- Current milestone plans completed: 0
- Current milestone phases completed: 0
- Average duration: n/a
- Total execution time: n/a

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 78. Matrix Contract And Claim Guardrails | 0/TBD | Not started | n/a |
| 79. Docker Matrix Selection And Preflight | 0/TBD | Not started | n/a |
| 80. uv And PyTorch ROCm Wheel Coordination | 0/TBD | Not started | n/a |
| 81. Runtime Evidence And Compatibility Reports | 0/TBD | Not started | n/a |
| 82. Validation Workflow, Docs, And CI Guardrails | 0/TBD | Not started | n/a |

**Recent Trend:**
- Last milestone: v1.17 shipped on 2026-05-26
- Trend: v1.18 starts with five planned phases for Docker-based ROCm version matrix evidence.

## Accumulated Context

### Decisions

- v1.18 starts at Phase 78 because v1.17 ended at Phase 77.
- v1.18 uses Target and Matrix Entry language for compatibility evidence.
- Illegal mixed-version Targets are blocked during preflight by default.
- Mixed-version debug override may continue probes or smoke only, without clean validation, score, paper-parity, or leaderboard claims.
- Docker container validation must be described as container ROCm user-space validation on recorded host driver/devices, never native host validation.

### Pending Todos

None found.

### Blockers/Concerns

- PyTorch ROCm wheel availability for ROCm 7.0.x and 7.2.x must be verified during Phase 80 planning/execution.
- Host-driver/container compatibility, Triton ROCm readiness, and image digest capture remain implementation research items for Phases 79-81.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Dataset extraction | Original paper 124-model / 235-problem extraction | Deferred | v1.10 scope |
| Hardware validation | MI300X, CDNA 3, and CDNA 4 real-hardware validation | Deferred | v1.10 scope |
| Native host matrix | Host reinstall or separate-host ROCm 7.0.x/7.1.x/7.2.x validation | Deferred | v1.18 scope |
| Public service | Hosted leaderboard or submission service | Deferred | v1.10 scope |

## Session Continuity

Last session: 2026-05-28
Stopped at: v1.18 roadmap created and ready for Phase 78 planning
Resume file: None
