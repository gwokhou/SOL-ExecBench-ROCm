---
gsd_state_version: 1.0
milestone: v1.16
milestone_name: ROCm Toolchain Research and Capability Routing
status: planning
last_updated: "2026-05-25T14:25:43.667Z"
last_activity: 2026-05-25
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-25)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.
**Current focus:** v1.16 ROCm toolchain research and capability routing

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-05-25 — Milestone v1.16 started

## Performance Metrics

**Velocity:**

- Current milestone plans completed: 0
- Current milestone phases completed: 0
- Average duration: n/a
- Total execution time: n/a

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 68. External ROCm Toolchain Research | 0 | - | - |
| 69. Toolchain Inventory And Lifecycle Model | 0 | - | - |
| 70. Capability Registry Schema | 0 | - | - |
| 71. Dynamic Probe And Routing Policy | 0 | - | - |
| 72. Toolchain Matrix Docs And Guardrails | 0 | - | - |

**Recent Trend:**

- Last milestone: v1.15 shipped on 2026-05-25
- Trend: v1.16 started to establish ROCm toolchain routing before static evidence

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

- Start Phase 68 with /gsd-discuss-phase 68 or /gsd-plan-phase 68
