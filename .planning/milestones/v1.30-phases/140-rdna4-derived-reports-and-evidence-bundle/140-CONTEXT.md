# Phase 140 Context

## Goal

Generate RDNA4 derived reports and evidence-bundle metadata from Phase 138
execution closure and Phase 139 timing evidence while preserving all claim
boundaries.

## Inputs

- Phase 138 execution closure:
  `out/rdna4-full-dataset/execution_closure.json`
- Phase 138 restored summary:
  `out/rdna4-full-dataset/run/summary.json`
- Phase 138 traces:
  `out/rdna4-full-dataset/run`
- Phase 139 environment evidence:
  `out/rdna4-timing-evidence/environment`
- Phase 139 timing evidence:
  `out/rdna4-timing-evidence/timing`
- Phase 139 stability report:
  `out/rdna4-timing-evidence/stability/evaluation-stability.json`

## Key Context

- Phase 138 completed bounded RDNA4 ready-subset execution, not full paper
  parity or leaderboard authority.
- Phase 139 found incomplete sudoers coverage for GPU clock lock/reset, and all
  timing sidecars selected PyTorch/device-event fallback rather than
  profiler-backed `rocprofv3` kernel activity timing.
- Phase 140 reports must treat timing as non-authoritative blocker/fallback
  evidence.
- Generated evidence under `out/` is local and should not be committed.
- Phase 140 derived sidecar generation includes real long-running and
  memory-heavy jobs. Run them as resumable per-problem jobs and poll
  infrequently; do not terminate healthy jobs solely because they run for many
  minutes or hours.
- To avoid dataset scripts exhausting memory/swap and taking the calling Codex
  session down with them, heavy derived jobs are run through
  `scripts/run_derived_isolated.py` using `systemd-run --user` with
  `MemoryMax=20G` and `MemorySwapMax=0`. Codex should poll logs/status instead
  of owning the heavy child process directly.
- Temporary derived-sidecar long-tail exclusions are recorded in
  `out/rdna4-derived-reports/rdna4-derived-long-tail-exclusions.json`. These
  exclusions are evidence-bundle context, not target features, and remain
  local/generated evidence until promoted through a later review.
