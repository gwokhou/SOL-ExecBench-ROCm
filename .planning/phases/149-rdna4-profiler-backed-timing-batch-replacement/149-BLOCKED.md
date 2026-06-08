# Phase 149 Blocker: Full 121-Problem Replacement Not Complete

## Status

Blocked for the full replacement objective.

## Blocker

The Phase 149 runner can collect real `rocprofv3` kernel trace artifacts, but
the first full-problem target,
`L1/002_vae_conv3x3_groupnorm_silu_residual_fused`, produced one
`INVALID_REFERENCE` workload under full profiler execution:

- `PASSED`: 19
- `INVALID_REFERENCE`: 1

The failing workload hit HIP out-of-memory on the 16 GiB RDNA4 validation GPU.
Because replacement sidecars now require every workload trace to be `PASSED`,
this problem cannot be counted as a full profiler-backed timing replacement on
the current host state.

## Evidence

- `out/rdna4-profiler-backed-timing-batch-real-full-smoke-v2/batch-summary.json`
- `out/rdna4-profiler-backed-timing-batch-real-full-smoke-v2/timing/L1/002_vae_conv3x3_groupnorm_silu_residual_fused.timing.json`
- `out/rdna4-profiler-backed-timing-batch-real-full-smoke-v2/coverage/coverage-summary.json`

## Next Actions

- Run broader batches to classify which of the 121 fallback problems can be
  replaced on the current 16 GiB RDNA4 host.
- Add skip/failure classification inputs if repeated long runs should avoid
  retrying known memory blockers first.
- Consider memory isolation or per-workload replacement semantics only if the
  claim boundary is updated; current Phase 149 semantics require full problem
  workload coverage.
