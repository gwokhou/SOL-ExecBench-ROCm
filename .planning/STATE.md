---
gsd_state_version: 1.0
milestone: v1.38
milestone_name: Upgrade SOL Evidence Contract for Confirmed Benchmark Claims
status: executing
stopped_at: Phase 192 verified
last_updated: "2026-07-09T09:00:00Z"
last_activity: 2026-07-09 -- Completed quick task 260709-spd: imported Superpowers docs into GSD planning
progress:
  total_phases: 5
  completed_phases: 3
  total_plans: 3
  completed_plans: 3
  percent: 60
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-21)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.
**Current focus:** Phase 193 — Measured Baseline Provenance and Coverage

## Current Position

Phase: 193 (Measured Baseline Provenance and Coverage) — READY FOR DISCUSSION
Plan: —
Status: Phase 192 verified; ready to discuss Phase 193
Last activity: 2026-07-09 -- Completed quick task 260709-spd: imported Superpowers docs into GSD planning

## Recent Trend

- v1.38 starts the confirmed evidence contract upgrade needed for HIP
  Playground cutover-gate decisions. It focuses on registered profiler
  artifacts, structured profile-summary evidence, official score evidence,
  measured baseline provenance, and HIP-facing confirmed evidence fixtures.

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
| 190 - Profiler Artifact Registration Closure | Make requested successful `rocprofv3` profile runs produce discoverable artifacts and citations | PROF-01, PROF-02, PROF-03 | Completed |
| 191 - Structured Profile Summary Evidence | Expand profile summary into structured profiling evidence with bottleneck hints while preserving diagnostic authority boundaries | PSUM-01, PSUM-02, PSUM-03 | Completed |
| 192 - Official Score Evidence Contract | Add official benchmark score evidence with score source, aggregation policy, and valid-run non-null score semantics | SCOR-01, SCOR-02, SCOR-03 | Completed |
| 193 - Measured Baseline Provenance and Coverage | Add measured baseline evidence and coverage validation for confirmed benchmark claims | BASE-01, BASE-02, BASE-03 | Pending |
| 194 - HIP Confirmed Evidence Package | Publish contract capabilities, fixtures, docs, and guardrails for HIP cutover-gate decisions | GATE-01, GATE-02, GATE-03 | Pending |

## Quick Tasks Completed

| Date | Task | Status | Notes |
|------|------|--------|-------|
| 2026-07-09 | 260709-spd-import-superpowers-docs | complete | Converted 25 Superpowers implementation plans into archived GSD quick artifacts, converted 7 design specs into `.planning/research/superpowers/`, and added an import index mapping every source file to its GSD artifact. |
| 2026-07-09 | 260709-my4-v1-43-tag-release-release-tag | complete | Updated the runtime SOL release constant and current sidecar fixtures/tests to `v1.43`; stale fixtures remain on `v1.42`; verified focused contract, feedback, profile-summary, and sidecar tests. |
| 2026-07-09 | 260709-tch-organize-platform-toolchain-modules | complete | Grouped `toolchain` platform modules into a subpackage while keeping the public facade; verified focused toolchain tests, Ruff, ty, and full pytest. |
| 2026-07-09 | 260709-evi-organize-runtime-evidence-modules | complete | Grouped `runtime_evidence` evidence modules into a subpackage while retaining `python -m` entry point; verified focused evidence/runtime tests, shell syntax, module help, Ruff, ty, and full pytest. |
| 2026-07-09 | 260709-plt-organize-platform-matrix-modules | complete | Grouped `dependency_matrix` and `docker_matrix` platform clusters into subpackages while retaining `python -m` entry points; verified focused platform tests, shell syntax, Ruff, ty, and full pytest. |
| 2026-07-09 | 260709-rpt-organize-report-modules | complete | Grouped clear `core/reports` feature clusters into subpackages and updated imports; verified focused report tests, Ruff, ty, and full pytest. |
| 2026-07-09 | 260709-dst-organize-dataset-modules | complete | Grouped clear `core/dataset` feature clusters into subpackages and updated imports; verified focused dataset tests, Ruff, ty, and full pytest. |
| 2026-07-09 | 260709-g1w-remove-remaining-backward-compatibility | complete | Removed remaining compatibility facades, legacy parser/fallback paths, stale CLI exports, and unused clock preset API; verified full pytest and Ruff. |
| 2026-07-09 | 260709-fjx-organize-flat-files-by-module-function | complete | Grouped flat bench and scoring feature clusters into subpackages while preserving compatibility facades; validation not run. |
| 2026-07-07 | 260707-s2p-refactor-repeated-helpers-to-use-standar | complete | Added shared JSON/model, text, and scoring parsing helpers; replaced repeated report serialization/checksum, Markdown, sha256, JSONL, ordered-unique, and subprocess text/tail logic; verified focused pytest and Ruff coverage. |
| 2026-07-03 | 260703-gze-close-the-sol-side-diagnostic-only-feedb | complete | Added HIP-facing sidecar identity aliases, documented profile-summary bottleneck hint ownership, prepared `v1.38-feedback-loop-rc1`, and verified focused pytest/Ruff coverage. |
| 2026-06-22 | 260622-eaa-fix-pre-push-ty-check-blockers-before-pu | complete | Fixed local `ty-check` failures blocking `git push origin main`; verified with `uv run ty check` and targeted pytest coverage. |
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

- v1.38 quick feedback-loop closure: HIP-facing sidecar freshness identity
  should prefer `sol_version`, `candidate_id`, and `source_sha256` while keeping
  `sol_contract_version`, `candidate_hash`, and `source_hash` as documented
  compatibility aliases.

- v1.38 quick feedback-loop closure: profiler-derived bottleneck hints remain
  in `profile_summary.sidecar.v1`; SOL does not duplicate them into
  `agent_feedback.items[]`.

### Pending Todos

- Start the next milestone with `$gsd-new-milestone`.
- Canonicalize SOL feedback/profile-summary sidecar identity fields around
  `candidate_id`, `source_sha256`, `sol_version`, and compatibility aliases;
  see `.planning/todos/pending/2026-07-03-canonicalize-feedback-sidecar-identity-fields.md`.
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

Last session: 2026-06-21T09:10:49.395Z
Stopped at: Phase 191 verified
Resume file: .planning/phases/192-official-score-evidence-contract

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone.
