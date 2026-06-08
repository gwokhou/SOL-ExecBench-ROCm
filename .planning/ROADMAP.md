# Roadmap: SOL ExecBench ROCm Port

## Milestones

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

**Status:** v1.31 completed and archived on 2026-06-08. Awaiting the next
milestone definition.

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
