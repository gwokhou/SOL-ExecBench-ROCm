---
status: complete
created_at: 2026-06-11T06:43:46Z
task: rdna4-v135-measurement-rerun
---

# RDNA4 v1.35 Measurement Rerun TODO

Goal: regenerate RDNA4 validation evidence after v1.35 measurement-accuracy
enhancements so timing, closure, derived evidence, and reports are same-source
and include PID lock, GPU isolation, clock audit, strict-isolation, stability
reason codes, and rocprofv3 overhead calibration evidence.

Constraints:
- Preserve the currently staged v1.36 rollback diff.
- Write new generated outputs under `out/rdna4-v135-rerun-20260611/`.
- Do not treat old `out/rdna4-full-dataset/execution_closure.json` as final
  authority because current readiness has drifted from that run.

TODO:
- [x] Rebuild current dataset `inventory.json`, `readiness.json`, and
  `ready_subset.json`.
- [x] Run RDNA4 profiler timing smoke to verify rocprofv3, clock, and
  environment sidecar viability.
- [x] Run rocprofv3 overhead calibration and save calibration sidecars.
- [x] Re-run RDNA4 execution closure from current readiness/ready subset.
- [x] Re-run profiler-backed timing batch with v1.35 isolation and calibration.
- [x] Re-run derived evidence generation from the new traces/timing sidecars.
- [x] Rebuild coverage, denominator, consistency, claim, trust, and bundle
  reports from same-source evidence.
- [x] Summarize completion status and any hardware/operator blockers.

Notes:
- 2026-06-11: Fixed calibration setup/teardown so the script locks clocks to
  `STABLE_PEAK` by default before strict-isolation measurement and only resets
  clocks when the script acquired the lock itself.
- 2026-06-11: Calibration completed at
  `out/rdna4-v135-rerun-20260611/rocprofv3-overhead-calibration.json` with
  `overhead_ms=0.001560`.
- 2026-06-11: Sandbox GPU execution produced false CPU-device errors; elevated
  PyTorch ROCm probe sees `AMD Radeon Graphics`, so RDNA4 GPU workloads must run
  with sandbox escalation on this host.
- 2026-06-11: Elevated `run_dataset.py --category L1 --limit 1` smoke wrote
  `execution_closure.gpu-smoke.json`; 12/16 workload shards passed and remaining
  failures were HIP OOM records, not CPU-device/sandbox errors.
- 2026-06-12: Derived rerun initially hit host OOM in
  `build_bound_graph()` because torch.fx shape propagation created real
  `torch.zeros(...)` CPU tensors for large workloads such as
  `L2/035_convnextv2_block_with_grn`; fixed by using meta tensors for tensor
  inputs and Python scalar values for scalar inputs.
- 2026-06-12: Full derived rerun completed with `--jobs 4`, maximum RSS about
  811 MiB, 61 problem summaries, 958 AMD score records, 978 AMD SOL sidecars,
  978 SOLAR sidecars, and execution closure `derived_evidence_missing=0`.
- 2026-06-12: Profiler timing batch OOM-killed the main batch process while
  aggregating `L2/008_moe_sparse_routing_and_dispatch`; the immediate cause was
  workload-sharded aggregation loading 16 slice sidecars of about 808 MB each
  and copying their full `evidence.parsed_rows` into a problem-level sidecar.
  Fixed aggregation to use compact manifest summaries
  (`kernel_activity_rows`/`kernel_duration_ms`) instead of re-reading full slice
  rows. Single-problem resume wrote an 18 KB compact `L2/008` aggregate and the
  full batch was restarted with `--resume`.
- 2026-06-12: Profiler batch then hit disk pressure from retained
  `workload-slices/` and `rocprofv3/` intermediates. Added default
  workload-slice compaction so completed slice sidecars retain compact kernel
  summaries, drop full parsed rows, and remove raw rocprofv3 run directories
  after manifest capture.
- 2026-06-12: Final profiler batch completed with exit code 0. Replacement
  timing count is 121, workload manifest count is 121, remaining resume targets
  are 0, and `batch-summary.json` reports `failed=0`, `fallback_or_missing=0`,
  `interrupted=false`, `profiler_blocked=0`, `profiler_backed=10`, and
  `partial_profiler_backed=6` for the 16 selected fallback targets.
- 2026-06-12: Rebuilt profiler timing coverage from the final replacement
  timing directory plus baseline timing evidence. Coverage summary is 235
  problems, 88 full profiler-backed, 28 partial profiler-backed, 0 fallback, 0
  profiler-blocked, and 73 ready-missing profiler timing problems.
- 2026-06-12: Rebuilt same-source paper denominator, AMD bound sanity,
  consistency, claim-upgrade, trust-summary, and prerelease artifact bundle
  reports under `out/rdna4-v135-rerun-20260611/`. Consistency now reports 0
  findings; the full prerelease bundle reports `overall_status=passed`.
- 2026-06-12: Fixed consistency drift detection so denominator
  `evidence_missing` on an attempted workload is not treated as a
  denominator/closure state conflict; this prevents failed attempted workloads
  that lack timing sidecars from being misclassified as stale blocked
  denominator rows.
- 2026-06-12: Updated related Markdown validation conclusions in
  `docs/CLAIMS.md`, `docs/rocm.md`, `docs/research_preview.md`,
  `docs/internal/RDNA4-AUTHORITY-GAP-CLOSURE.md`, and
  `docs/internal/RDNA4-DENOMINATOR-POLICY.md` to reflect the v1.35 rerun
  evidence while preserving `full_profiler_backed_timing_coverage=false` and
  non-authority claim boundaries.
