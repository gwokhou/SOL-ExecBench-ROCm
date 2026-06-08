# Roadmap: SOL ExecBench ROCm Port

## Milestones

- Active **v1.32 RDNA4 Profiler Timing Coverage Closure** - Phases 148-150.
  See `.planning/milestones/v1.32-ROADMAP.md`.
- Complete **v1.31 RDNA4 Follow-Up Evidence Hardening** - Phases 142-147
  (shipped 2026-06-08). See `.planning/milestones/v1.31-ROADMAP.md`.
- Complete **v1.30 RDNA4 Benchmark-Grade Validation Closure** - Phases 136-141
  (shipped 2026-06-08). See `.planning/milestones/v1.30-ROADMAP.md`.
- Complete **v1.29 Dataset Migration and Compliance** - Phases 131-135
  (shipped 2026-06-04). See `.planning/milestones/v1.29-ROADMAP.md`.
- Complete **v1.28 CDNA3 Test and Documentation Readiness** - Phases 127-130
  (shipped 2026-06-04). See `.planning/milestones/v1.28-ROADMAP.md`.
- Complete **v1.27 Copyright Provenance Cleanup** - Phases 123-126
  (shipped 2026-06-02). See `.planning/milestones/v1.27-ROADMAP.md`.
- Complete **v1.26 Public Prerelease and Research Preview** - Phases 119-122
  (shipped 2026-06-02). See `.planning/milestones/v1.26-ROADMAP.md`.
- Complete **v1.25 Engineering Prerelease** - Phases 114-118
  (shipped 2026-06-01). See `.planning/milestones/v1.25-ROADMAP.md`.
- Earlier milestones are archived under `.planning/milestones/`.

## Current Position

**Status:** v1.32 active. Phase 150 complete; Phase 149 full replacement remains blocked.

### Phase 148: RDNA4 Profiler-Backed Timing Coverage Closure

**Status:** Complete

**Goal:** Account for profiler-backed `rocprofv3` timing coverage across the
full 235-problem denominator before attempting expanded RDNA4 timing batches.

**Requirements:** RDNA4-PROF-COV-01, RDNA4-PROF-COV-02

**Deliverables:**
- Problem-denominator profiler timing coverage report with explicit
  profiler-backed, fallback, ready-missing, and readiness-blocked states.
- Scripted report generation over `data/SOL-ExecBench/benchmark` and existing
  timing sidecars.
- CPU-safe tests preserving the claim boundary that fallback timing is not
  profiler-backed coverage.

### Phase 149: RDNA4 Profiler-Backed Timing Batch Replacement

**Status:** Blocked 2026-06-08.

**Goal:** Replace the 121 RDNA4 fallback timing sidecars with explicit
`rocprofv3` profiler-backed kernel activity timing sidecars where the current
coverage ledger marks problems as `timing_fallback`.

**Requirements:** RDNA4-PROF-COV-03, RDNA4-PROF-COV-04

**Deliverables:**
- Batch runner that profiles staged `eval_driver.py` directly for fallback
  timing problems and writes replacement timing sidecars.
- Resume/limit controls so long RDNA4 timing jobs can be run incrementally.
- CPU-safe tests covering target selection, forced profiler policy, and
  no-fallback replacement semantics.

### Phase 150: RDNA4 Profiler Replacement Classification

**Status:** Complete

**Goal:** Convert the Phase 149 all-or-nothing replacement blocker into an
auditable profiler replacement classification ledger across the same
235-problem denominator.

**Requirements:** RDNA4-PROF-COV-05, RDNA4-PROF-COV-06

**Deliverables:**
- Coverage statuses and totals for `partial_profiler_backed` and
  `profiler_blocked` replacement attempts.
- Batch metadata that records replacement status, workload counts, trace status
  counts, and failure reason without counting partial evidence as complete
  profiler-backed timing coverage.
- Resume behavior that can skip already classified full-problem attempts while
  still allowing workload-limited smoke artifacts to be retried.

## Active Guardrails

- RDNA4 validation jobs may run for many hours. Poll, checkpoint, and preserve
  logs/artifacts; do not terminate healthy long-running processes solely due to
  duration.
- Long RDNA4 derived/dataset jobs that can exhaust memory must run through
  `scripts/run_derived_isolated.py --launch-mode systemd` or equivalent
  transient `systemd-run --user` units with memory/swap caps. Codex should poll
  status/log files rather than owning heavy child processes.
- Temporarily excluded long-tail shards/problems/workloads must be explicit,
  auditable, reversible, and visible in denominator and closure reports.
- CDNA3/MI300X hardware validation remains separate until complete exact
  MI300X evidence is archived; hardware validation remains deferred for MI300X
  until that evidence exists, and existing `requires_cdna3` readiness coverage
  remains a separate marker/test surface rather than a validation claim.
- CDNA4 validation remains deferred until suitable hardware evidence exists.
