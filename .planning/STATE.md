---
gsd_state_version: 1.0
milestone: v1.18
milestone_name: ROCm Version Matrix via Docker
status: ready_to_plan
stopped_at: Phase 78 complete (2/2) — ready to discuss Phase 79
last_updated: 2026-05-28T05:51:54.594Z
last_activity: 2026-05-28
progress:
  total_phases: 5
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
  percent: 20
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-28)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.
**Current focus:** Phase 79 — docker matrix selection and preflight

## Current Position

Phase: 79
Plan: Not started
Status: Ready to plan
Last activity: 2026-05-28

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**

- Current milestone plans completed: 2
- Current milestone phases completed: 1
- Average duration: 5.5min
- Total execution time: 11min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 78. Matrix Contract And Claim Guardrails | 2/2 | 11min | 5.5min |
| 79. Docker Matrix Selection And Preflight | 0/TBD | Not started | n/a |
| 80. uv And PyTorch ROCm Wheel Coordination | 0/TBD | Not started | n/a |
| 81. Runtime Evidence And Compatibility Reports | 0/TBD | Not started | n/a |
| 82. Validation Workflow, Docs, And CI Guardrails | 0/TBD | Not started | n/a |
| 78 | 2 | - | - |

**Recent Trend:**

- Last milestone: v1.17 shipped on 2026-05-26
- Trend: v1.18 starts with five planned phases for Docker-based ROCm version matrix evidence.

| Phase 78 P1 | 5min | 2 tasks | 2 files |
| Phase 78 P2 | 6min | 3 tasks | 6 files |

## Accumulated Context

### Decisions

- v1.18 starts at Phase 78 because v1.17 ended at Phase 77.
- v1.18 uses Target and Matrix Entry language for compatibility evidence.
- Illegal mixed-version Targets are blocked during preflight by default.
- Mixed-version debug override may continue probes or smoke only, without clean validation, score, paper-parity, or leaderboard claims.
- Docker container validation must be described as container ROCm user-space validation on recorded host driver/devices, never native host validation.
- [Phase 78]: Target/requested values and observed host, container, Python dependency, toolchain, and GPU evidence are modeled as separate required objects.
- [Phase 78]: Score, paper-parity, and leaderboard authority are literal false fields on every Matrix Entry claim boundary.
- [Phase 78]: Compatibility Matrix Entries are strict sidecars separate from canonical trace, timing, scoring, and benchmark result schemas.
- [Phase 78]: Mixed-version Matrix Entries are blocked before benchmark execution by default.
- [Phase 78]: Docker/container scoped Matrix Entries cannot claim host_validated or native_host_validated=true.
- [Phase 78]: Explicit mixed-version debug override allows probes and smoke execution only, while clean validation and authority flags remain false.
- [Phase 78]: Compatibility matrix evidence remains sidecar-only and absent from canonical Definition, Workload, and Trace payloads.

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

Last session: 2026-05-28T05:38:04.175Z
Stopped at: Completed 78-02-PLAN.md
Resume file: None
