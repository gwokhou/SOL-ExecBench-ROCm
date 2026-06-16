---
gsd_state_version: 1.0
milestone: v1.36
milestone_name: SOL Agent Feedback Sidecar Producer
status: Awaiting next milestone
stopped_at: v1.36 milestone completion
last_updated: "2026-06-16T03:17:09.338Z"
last_activity: 2026-06-16 — Milestone v1.36 completed and archived
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 6
  completed_plans: 6
  percent: 100
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-16)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.
**Current focus:** Planning next milestone

## Current Position

Phase: Milestone v1.36 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-06-16 — Milestone v1.36 completed and archived

## Recent Trend

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
| 181 - Feedback Contract and Capability Surface | Optional feedback/profile-summary capabilities and documentation boundaries without canonical trace drift | CNTR-01, CNTR-02, CNTR-03 | Complete |
| 182 - Diagnostic Sidecar Schema and Generator | Strict `sol_execbench.agent_feedback.v1` schema and `trace.jsonl.agent-feedback.json` persistence | SIDE-01, SIDE-02, SIDE-03, SIDE-04 | Complete |
| 183 - Freshness Identity and Artifact References | Trace/run/candidate identity plus compact artifact citations for stale-feedback detection | IDEN-01, IDEN-02, IDEN-03 | Complete |
| 184 - Governance Guardrails and Compatibility Fixtures | Diagnostic-only authority validation and release/claim guardrails | GOVR-01, GOVR-02, GOVR-03 | Complete |
| 185 - HIP Consumer Integration Package and Docs | HIP-facing fixtures, examples, mapping notes, and deterministic fixture tests | FIXT-01, FIXT-02, FIXT-03 | Complete |

## Quick Tasks Completed

| Date | Task | Status | Notes |
|------|------|--------|-------|
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

### Pending Todos

- Start the next milestone with `$gsd-new-milestone`.
- Coordinate the generated sidecar fixture shape with HIP Playground Phase 141
  before HIP adapter implementation begins.

### Blockers/Concerns

- Stale sidecar identity is a primary risk when trace paths are reused across
  retries or resumed runs.

- Raw profiler dumps, trace rows, source text, or temporary absolute paths must
  not leak into prompt-facing feedback summaries.

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
| Quick task | 260531-rdf-add-run-dataset-closure-e2e-gaps | completed |
| Quick task | 260531-uki-add-remaining-requires-rocm-e2e-coverage | completed |
| Quick task | 260613-44p-complete-short-term-rdna4-profiler-backe | unknown |
| Quick task | 260613-fix-l2041-partial | missing |
| Quick task | 260613-iut-close-rdna4-ready-missing-profiler-timin | missing |
| Quick task | 260613-lia-close-remaining-29-rdna4-l2-ready-missin | unknown |
| Quick task | 260613-try-fix-rdna4-l2-profiler-blockers | unknown |

## Session Continuity

Last session: 2026-06-16T03:17:09Z
Stopped at: v1.36 milestone completion
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
