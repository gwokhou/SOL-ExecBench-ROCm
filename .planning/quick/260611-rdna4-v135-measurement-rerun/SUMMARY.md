---
status: complete
completed_at: 2026-06-12T16:12:23Z
task: rdna4-v135-measurement-rerun
---

# RDNA4 v1.35 Measurement Rerun Summary

Completed the RDNA4 v1.35 rerun evidence package under
`out/rdna4-v135-rerun-20260611/`.

Key outcomes:
- Execution closure rebuilt from current readiness/ready subset with
  `derived_evidence_missing=0`.
- Derived evidence generation completed with bounded host memory after moving
  FX shape propagation sample inputs to meta tensors.
- Profiler timing batch completed with 121 replacement timing sidecars, 121
  workload manifests, and 0 remaining resume targets.
- Profiler batch summary reports `failed=0`, `fallback_or_missing=0`,
  `interrupted=false`, and `profiler_blocked=0`.
- Coverage, denominator, consistency, claim, trust, and prerelease bundle
  reports were rebuilt from the rerun evidence. Consistency reports 0 findings;
  the full prerelease bundle reports `overall_status=passed`.
- Related Markdown validation conclusions were updated in `docs/CLAIMS.md`,
  `docs/rocm.md`, `docs/research_preview.md`,
  `docs/internal/RDNA4-AUTHORITY-GAP-CLOSURE.md`, and
  `docs/internal/RDNA4-DENOMINATOR-POLICY.md`.

Fixes made during the rerun:
- Avoided derived host OOM by using meta tensors during AMD bound graph shape
  propagation.
- Avoided workload-sharded profiler aggregation OOM by aggregating from compact
  manifest summaries instead of loading full per-slice parsed rows.
- Avoided profiler intermediate ENOSPC by compacting completed workload slices
  and removing raw rocprofv3 run directories after manifest capture.
- Fixed consistency drift detection so an evidence gap on an attempted workload
  is not treated as a blocked-denominator conflict.

Residual boundaries:
- Full profiler-backed timing coverage is still false: coverage is 88 full
  profiler-backed problems and 28 partial profiler-backed problems out of the
  235-problem denominator.
- The evidence remains diagnostic/review evidence, not paper parity,
  leaderboard authority, or broad hardware validation.
