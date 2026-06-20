---
gsd_state_version: 1.0
milestone: v1.37
milestone_name: Profile Summary Sidecar v1
status: awaiting_next_milestone
last_updated: "2026-06-20T13:21:10.117Z"
last_activity: 2026-06-20
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 4
  completed_plans: 4
  percent: 100
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-16)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.
**Current focus:** No active milestone

## Current Position

Phase: —
Plan: —
Status: v1.37 shipped; awaiting next milestone
Last activity: 2026-06-16 — v1.37 profile-summary sidecar milestone implemented, audited, and archived

## Recent Trend

- v1.37 starts the focused follow-up to v1.36 by turning
  `profile_summary.sidecar.v1` from a reserved optional capability into a real
  diagnostic sidecar artifact with strict schema, producer wiring, freshness
  identity, citations, governance, and HIP-facing fixtures/docs.

- v1.37 implementation is complete. SOL now emits optional
  `<trace>.profile-summary.json` diagnostic sidecars with strict
  `sol_execbench.profile_summary.v1` schema, compact citations, freshness and
  governance validation helpers, and HIP-facing fixture/docs coverage.

- v1.37 shipped on 2026-06-16. Active requirements were archived to
  `.planning/milestones/v1.37-REQUIREMENTS.md`, roadmap details were archived
  to `.planning/milestones/v1.37-ROADMAP.md`, and the audit is recorded at
  `.planning/milestones/v1.37-MILESTONE-AUDIT.md`.

- v1.36 shipped the five-phase SOL-side producer milestone for HIP Playground
  v1.26. SOL now emits optional diagnostic agent-feedback sidecars with
  contract capability tokens, strict schema/writer support, freshness identity,
  compact artifact citations, governance guardrails, and HIP-facing fixtures.

- v1.35 shipped on 2026-06-11. 6 phases, 7 plans, 23 requirements satisfied.
  Added PID locks, timing isolation, CPU-parallel staging, parallel dispatch,
  evaluation stability reason codes, GPU device isolation, strict-isolation
  mode, and rocprofv3 overhead calibration.

- v1.34 shipped on 2026-06-09. Phases 170-174 reduced RDNA4 readiness_blocked
  from 114 to 59 over a stable 235-problem denominator.

## Phase Map

| Phase | Goal | Requirements | Status |
|-------|------|--------------|--------|
| 186 - Profile Summary Contract and Schema | Concrete optional `sol_execbench.profile_summary.v1` contract, strict schema, and authority boundary | PCON-01, PCON-02, PCON-03, PSCH-01, PSCH-02, PSCH-03 | Complete |
| 187 - Profile Summary Producer and CLI Persistence | Persist `<trace>.profile-summary.json` without changing trace, profile, static evidence, or agent-feedback behavior | PROD-01, PROD-02, PROD-03 | Complete |
| 188 - Profile Summary Freshness and Governance | Identity, artifact citations, stale-state validation, and diagnostic-only authority guardrails | PGOV-01, PGOV-02, PGOV-03 | Complete |
| 189 - HIP Profile Summary Fixtures and Docs | HIP-facing fixtures, mapping docs, and deterministic CPU-safe tests | PFIX-01, PFIX-02, PFIX-03 | Complete |

## Quick Tasks Completed

| Date | Task | Status | Notes |
|------|------|--------|-------|
| 2026-06-20 | 260620-tnn-cherry-pick-local-backup-changes-to-main | complete | Cherry-picked local backup commit `a33600b` onto current `main`, preserving current codebase mapping state while restoring wrapper scripts and documentation/evidence notes. |
| 2026-06-16 | 260616-n3b-record-deferred-hip-v1-26-feedback-sidec | complete | Recorded `profile_summary.sidecar.v1` producer/schema and profiler-counter-derived bottleneck diagnostics as deferred follow-ups. |
| 2026-06-16 | 260616-n06-fix-sol-agent-feedback-source-hash-and-c | complete | Filled SOL CLI agent-feedback `source_hash` from `Solution.hash()` and documented `candidate_hash` as solution-label identity, with targeted pytest/Ruff verification. |
| 2026-06-16 | 260616-mpd-close-hip-v1-26-sol-feedback-sidecar-qui | complete | Clarified reserved profile-summary capability, filled feedback identity fields from emitted trace data, and closed SOL bottleneck vocabulary with targeted pytest/Ruff verification. |
| 2026-06-12 | rdna4-v135-measurement-rerun | complete | Rebuilt RDNA4 v1.35 rerun closure, derived evidence, profiler timing batch, coverage, denominator, consistency, claim, trust, and bundle reports under `out/rdna4-v135-rerun-20260611/`; fixed derived/profiler OOM, profiler ENOSPC, and consistency evidence-gap drift misclassification. |

## Accumulated Context

### Decisions

- fcntl.flock selected over PID-in-file locking for kernel-managed auto-release.
- ThreadPoolExecutor chosen over ProcessPoolExecutor due to torch fork-safety.
- CPU-parallel staging + GPU-serial profiling architecturally enforced.
- Overhead calibration uses inner subprocess under rocprofv3.
- v1.36: Agent-feedback/profile-summary sidecars are optional diagnostic
  sidecars only; canonical Trace JSONL remains the authority for correctness,
  timing, scoring, and evaluation status.

- v1.36: HIP Playground owns `ProfileDigest`, adapter normalization, strategy
  hints, and runtime prompt assembly; SOL owns sidecar schema, generation,
  freshness identity, citations, and authority guardrails.

- v1.36: Feedback may guide a next experiment but cannot promote evidence tier,
  score authority, confirmed improvement, release gates, cutover eligibility,
  paper parity, or leaderboard readiness.

- v1.37: Profile-summary sidecars may inform HIP adapter diagnostics, but cannot
  promote correctness, timing, performance, score, evidence-tier, release-gate,
  cutover, paper-parity, leaderboard, or claim-upgrade authority.

### Pending Todos

- Start the next milestone with `$gsd-new-milestone`.
- Keep profiler-counter-derived bottleneck diagnostics deferred unless the next
  milestone explicitly scopes hardware/counter taxonomy work.

### Blockers/Concerns

- Full `uv run pytest tests/` remains blocked on this Mac host by broader
  environment/repo issues: missing `triton`, missing legacy report scripts, and
  Docker shell compatibility failures.

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
| Agent feedback diagnostics | Profiler-counter-derived bottleneck diagnostics for occupancy, registers, LDS, bandwidth, cache, and utilization | Deferred |
| Quick task | 260531-rdf-add-run-dataset-closure-e2e-gaps | completed |
| Quick task | 260531-uki-add-remaining-requires-rocm-e2e-coverage | completed |
| Quick task | 260613-44p-complete-short-term-rdna4-profiler-backe | unknown |
| Quick task | 260613-fix-l2041-partial | missing |
| Quick task | 260613-iut-close-rdna4-ready-missing-profiler-timin | missing |
| Quick task | 260613-lia-close-remaining-29-rdna4-l2-ready-missin | unknown |
| Quick task | 260613-try-fix-rdna4-l2-profiler-blockers | unknown |

## Session Continuity

Last session: 2026-06-16T03:17:09Z
Stopped at: v1.37 milestone planning complete
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone.
