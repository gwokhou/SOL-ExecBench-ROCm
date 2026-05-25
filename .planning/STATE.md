---
gsd_state_version: 1.0
milestone: v1.16
milestone_name: ROCm Toolchain Research and Capability Routing
status: awaiting-next-milestone
stopped_at: v1.16 archived
last_updated: "2026-05-25T15:10:00.000Z"
last_activity: 2026-05-25 — Milestone v1.16 completed and archived
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 5
  completed_plans: 5
  percent: 100
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-25)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.
**Current focus:** v1.16 archived; awaiting next milestone

## Current Position

Phase: Milestone v1.16 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-05-25 — Milestone v1.16 completed and archived

## Performance Metrics

**Velocity:**

- Current milestone plans completed: 5
- Current milestone phases completed: 5
- Average duration: n/a
- Total execution time: n/a

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 68. External ROCm Toolchain Research | 1 | complete | n/a |
| 69. Toolchain Inventory And Lifecycle Model | 1 | complete | n/a |
| 70. Capability Registry Schema | 1 | complete | n/a |
| 71. Dynamic Probe And Routing Policy | 1 | complete | n/a |
| 72. Toolchain Matrix Docs And Guardrails | 1 | complete | n/a |

**Recent Trend:**

- Last milestone: v1.16 shipped on 2026-05-25
- Trend: toolchain routing foundation complete; next likely work is v1.17
  Static Kernel Evidence.

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md. Recent decisions affecting current work:

- v1.16 prioritizes ROCm toolchain research and capability routing before Static Kernel Evidence.
- ROCm tool availability must be modeled by hardware generation, GPU architecture, ROCm version, artifact type, and evidence level.
- Static Kernel Evidence is explicitly deferred to v1.17.
- Routing decisions must not be treated as correctness, performance, paper-parity, or leaderboard authority.
- Detailed historical decisions are preserved in `.planning/PROJECT.md` and archived milestone artifacts.

### Pending Todos

None found.

### Blockers/Concerns

None active.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260524-xb3 | Fix PR base and prepare SOL evaluator contract branch for merge | 2026-05-24 | 5d4892d | [260524-xb3-fix-pr-base-and-prepare-sol-evaluator-co](./quick/260524-xb3-fix-pr-base-and-prepare-sol-evaluator-co/) |
| 260525-097 | Backfill v1.12 GSD milestone for evaluator contract PR | 2026-05-25 | this commit | [260525-097-backfill-v1-12-gsd-milestone-for-evaluat](./quick/260525-097-backfill-v1-12-gsd-milestone-for-evaluat/) |
| 260525-ruff | Configure Ruff as a dev dependency | 2026-05-25 | this commit | [260525-configure-ruff-dev-dependency](./quick/260525-configure-ruff-dev-dependency/) |
| 260525-ty | Configure Ty as a dev dependency | 2026-05-25 | this commit | [260525-configure-ty-dev-dependency](./quick/260525-configure-ty-dev-dependency/) |
| 260525-fix-ty-s0-s1 | Fix Ty S0/S1 diagnostics | 2026-05-25 | this commit | [260525-fix-ty-s0-s1](./quick/260525-fix-ty-s0-s1/) |
| 260525-q95 | Add ROCm device-node-aware skip reasons for Codex sandboxed test runs | 2026-05-25 | this commit | [260525-q95-add-rocm-device-node-aware-skip-reasons-](./quick/260525-q95-add-rocm-device-node-aware-skip-reasons-/) |
| 260525-qft | Add GitHub Actions code-quality workflow matching hip-playground | 2026-05-25 | this commit | [260525-qft-add-github-actions-code-quality-workflow](./quick/260525-qft-add-github-actions-code-quality-workflow/) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Dataset extraction | Original paper 124-model / 235-problem extraction | Deferred | v1.10 scope |
| Hardware validation | MI300X, CDNA 3, and CDNA 4 real-hardware validation | Deferred | v1.10 scope |
| Hardware validation | NVFP4 and MXFP4 validation | Deferred | v1.10 scope |
| Public service | Hosted leaderboard or submission service | Deferred | v1.10 scope |
| Claims | NVIDIA Blackwell/B200 equivalence | Deferred | v1.10 scope |

## Session Continuity

Last session: 2026-05-23T11:26:49.344Z
Stopped at: Completed 52-03-PLAN.md
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
