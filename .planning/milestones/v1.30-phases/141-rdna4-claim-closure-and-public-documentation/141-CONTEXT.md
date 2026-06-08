# Phase 141 Context

## Goal

Close public RDNA4 wording against the Phase 138-140 evidence set without
turning bounded evidence into paper parity, timing authority, leaderboard
authority, or unrelated hardware validation claims.

## Inputs

- Phase 138 execution closure:
  `out/rdna4-full-dataset/execution_closure.json`
- Phase 138 restored run summary:
  `out/rdna4-full-dataset/run/summary.json`
- Phase 139 timing/stability evidence:
  `out/rdna4-timing-evidence/`
- Phase 140 derived evidence bundle:
  `out/rdna4-derived-reports/bundle/evidence-bundle.json`
- Phase 140 summary and verification docs.

## Key Context

- RDNA4 `gfx1200` evidence is bounded to the ready subset that was actually
  selected and executed: 121 ready problems, 1907 attempted workloads, 1761
  passed workloads, 146 failed workloads, 86 OK problems, 35 FAIL problems, and
  12 explicit `missing_trace` workload records.
- Derived evidence exists for 1895 score records: 172 scored and 1723 unscored.
- AMD SOL v2 and SOLAR derivation sidecar coverage is complete only after 56
  temporary long-tail sidecar exclusions are applied: 1839 traces have both
  sidecars and 0 unexcluded sidecars are missing.
- Timing remains non-authoritative. Clock lock/reset sudoers coverage is
  incomplete and Phase 139 timing sidecars used PyTorch/device-event fallback
  rather than profiler-backed `rocprofv3` kernel activity timing.
- Phase 140 consistency/trust reports preserve blocker findings rather than
  upgrading claim authority.
- Repeated OOM/swap exhaustion can crash the caller if heavy dataset scripts
  are launched directly from Codex. Future long derived/dataset jobs should use
  `scripts/run_derived_isolated.py --launch-mode systemd` or equivalent
  transient `systemd-run --user` units with `MemoryMax` and `MemorySwapMax`;
  Codex should poll status/log files instead of owning memory-heavy children.

