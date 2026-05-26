---
gsd_state_version: 1.0
milestone: v1.17
milestone_name: Static Kernel Evidence
status: Awaiting next milestone
stopped_at: Milestone v1.17 completed, audited, and archived
last_updated: "2026-05-26T02:41:08.490Z"
last_activity: 2026-05-26 — Milestone v1.17 completed and archived
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 5
  completed_plans: 5
  percent: 100
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-26)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.
**Current focus:** Milestone complete

## Current Position

Phase: Milestone v1.17 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-05-26 — Milestone v1.17 completed and archived

## Performance Metrics

**Velocity:**

- Current milestone plans completed: 5
- Current milestone phases completed: 5
- Average duration: n/a
- Total execution time: n/a

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 73. Static Evidence Contract And Guardrails | 1 | complete | n/a |
| 74. Build Artifact Discovery And Manifest | 1 | complete | n/a |
| 75. Routed Static Extractor Adapters | 1 | complete | n/a |
| 76. CLI Sidecar Integration And Reports | 1 | complete | n/a |
| 77. Documentation, Guardrails, And Live Validation | 1 | complete | n/a |

**Recent Trend:**

- Last milestone: v1.16 shipped on 2026-05-25
- Trend: v1.17 shipped five phases from contract to discovery, routed
  extractors, CLI/reporting, then docs/live validation.

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md. Recent decisions affecting current work:

- v1.16 prioritizes ROCm toolchain research and capability routing before Static Kernel Evidence.
- v1.17 starts at Phase 73 because v1.16 ended at Phase 72.
- Static Kernel Evidence remains opt-in, sidecar-only, and diagnostic.
- Static extraction must use v1.16 routing, not ad hoc executable lookup.
- Static evidence must not mutate canonical trace JSONL, correctness, timing,
  scoring, default benchmark behavior, paper-parity, or leaderboard claims.

- Detailed historical decisions are preserved in `.planning/PROJECT.md` and archived milestone artifacts.

### Pending Todos

None found.

### Blockers/Concerns

The bounded RDNA 4 run collected static evidence successfully, but benchmark
correctness did not pass for the RMSNorm example. CDNA 3, CDNA 4, Triton,
RGA-rich resource parsing, and paper-scale static coverage remain deferred
unless direct evidence is produced in a later milestone.

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
| Compatibility validation | ROCm 7.0.x/7.1.x/7.2.x by GPU-generation validation matrix | Deferred | v1.16 follow-up discussion |
| Hardware validation | NVFP4 and MXFP4 validation | Deferred | v1.10 scope |
| Public service | Hosted leaderboard or submission service | Deferred | v1.10 scope |
| Claims | NVIDIA Blackwell/B200 equivalence | Deferred | v1.10 scope |
| Quick task metadata | 260524-xb3-fix-pr-base-and-prepare-sol-evaluator-co | Missing completion metadata acknowledged | v1.17 close audit |
| Quick task metadata | 260525-097-backfill-v1-12-gsd-milestone-for-evaluat | Missing completion metadata acknowledged | v1.17 close audit |
| Quick task metadata | 260525-audit-fix-remaining-suite-reds | Missing completion metadata acknowledged | v1.17 close audit |
| Quick task metadata | 260525-configure-ruff-dev-dependency | Missing completion metadata acknowledged | v1.17 close audit |
| Quick task metadata | 260525-configure-ty-dev-dependency | Missing completion metadata acknowledged | v1.17 close audit |
| Quick task metadata | 260525-fix-ty-s0-s1 | Missing completion metadata acknowledged | v1.17 close audit |
| Quick task metadata | 260525-q95-add-rocm-device-node-aware-skip-reasons- | Missing completion metadata acknowledged | v1.17 close audit |
| Quick task metadata | 260525-qft-add-github-actions-code-quality-workflow | Missing completion metadata acknowledged | v1.17 close audit |

## Session Continuity

Last session: 2026-05-23T11:26:49.344Z
Stopped at: Milestone v1.17 completed, audited, and archived
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
