---
status: complete
quick_id: 260613-close-6-profiler-blocked
slug: close-6-profiler-blocked
completed_at: 2026-06-13T20:36:00+08:00
---

# Summary

Completed validation for the 6 remaining `profiler_blocked` RDNA4 targets under
the agreed criterion: if the problem cannot execute on the current machine, a
concrete local resource-limit conclusion counts as completed validation.

Result:

- Coverage does not increase on this host.
- The 6 targets remain non-profiler-backed because the current ROCm-visible GPU
  has only `15.92 GiB` VRAM.
- Existing manifests show timing-pool peak estimates from `24.04 GiB` to
  `48.13 GiB`.
- Two direct probes confirmed real GPU input-generation OOM, not stale import or
  rocprofv3-only failure:
  - `L2/024_moe_expert_parallel_execution`, workload offset `0`
  - `L2/012_moe_expert_batched_execution_with_capacity_factor`, workload offset
    `0`

Probe evidence:

- `out/rdna4-close-6-profiler-blocked-20260613/probe-l2024-offset0/timing/L2/024_moe_expert_parallel_execution.timing.json`
- `out/rdna4-close-6-profiler-blocked-20260613/probe-l2012-offset0/timing/L2/012_moe_expert_batched_execution_with_capacity_factor.timing.json`

Conclusion:

- These 6 targets require a larger-VRAM RDNA4/ROCm-visible GPU to produce
  complete profiler-backed timing without changing benchmark semantics.
- No merged coverage promotion was performed because the new evidence is
  negative `RUNTIME_ERROR`/`gen_inputs_oom_blocked` evidence.
