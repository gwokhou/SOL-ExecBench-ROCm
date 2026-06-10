---
gsd_state_version: 1.0
milestone: v1.35
milestone_name: Script Parallelism and Safety Hardening
status: planning
last_updated: "2026-06-10T14:00:00.827Z"
last_activity: 2026-06-10
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-09)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.
**Current focus:** Planning next milestone

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-06-10 — Milestone v1.35 started

## Recent Trend

- v1.34 shipped on 2026-06-09. Phases 170-174 reduced RDNA4 readiness_blocked
  from 114 to 59 over a stable 235-problem denominator. All 55 custom-input
  blockers promoted to ready. Quant false-positive hints reclassified.
  FlashInfer workloads semantically split. Claim guardrails preserved.

- v1.33 shipped on 2026-06-09 with RDNA4 benchmark-grade evidence closure
  including denominator policy hardening, coverage recompute, profiler timing
  closure, clock-lock evidence, and authority-gap boundaries.

- v1.30 shipped on 2026-06-08 with bounded RDNA4 `gfx1200` validation closure
  (121 ready problems, 1907 attempted workloads, 1761 passed, 86 OK, 35 FAIL).

## Accumulated Context

### Decisions

- v1.34 readiness blocker reduction is not validation success unless execution
  evidence passes.

- RDNA4 timing remains bounded: STABLE_PEAK gives near-maximum evidence but is
  not benchmark-grade authority.

- CDNA3/MI300X, CDNA4, paper-scale, leaderboard, hosted-service, hard-sandbox,
  and dataset-redistribution boundaries remain deferred.

### Pending Todos

- Define and start the next milestone with `/gsd-new-milestone`.

### Blockers/Concerns

- `out/rdna4-coverage-current/coverage.json` has stale FlashInfer bucket codes
  for 018/019 problems (code is correct, artifact needs regeneration).

- RDNA4 timing remains non-authoritative pending clock-lock stability evidence.

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
| quick_task | 260531-rdf-add-run-dataset-closure-e2e-gaps | completed |
| quick_task | 260531-uki-add-remaining-requires-rocm-e2e-coverage | completed |
| quick_task | 260602-mqi-fix-stale-project-configuration-audit-fi | resolved |
| quick_task | 260602-mqr-fix-second-pass-configuration-audit-findings | resolved |
| quick_task | 260602-msd-unwrap-readme-prose-lines | resolved |
| debug | 260607-remote-ci-failure | resolved |
| quick_task | 260604-vjx-scripts-benchmark | resolved |
| quick_task | 260605-port-nvfp4-reference-scaled-mm | resolved |
| quick_task | 260606-clarify-cdna3-mi300x-hierarchy | resolved |
| quick_task | 260606-clarify-mi308x-cdna3-validation-docs | resolved |
| quick_task | 260606-wis-fix-current-pytest-failures-after-codeba | resolved |

## Session Continuity

Last session: 2026-06-09T23:30:00.000Z
Stopped at: Milestone v1.34 shipped and archived
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
