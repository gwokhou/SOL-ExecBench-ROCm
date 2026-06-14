---
status: completed
quick_id: 260613-close-6-profiler-blocked
slug: close-6-profiler-blocked
description: Close the 6 remaining ready profiler_blocked RDNA4 targets
created_at: 2026-06-13T20:30:19+08:00
---

# Quick Task 260613-close-6-profiler-blocked

## Goal

Close the 6 remaining `profiler_blocked` problems from the latest canonical
RDNA4 merged artifacts.

Baseline:

- Canonical merged directory:
  `out/rdna4-validation-reeval-20260613-latest-plus-l2041/merged/`
- Current coverage:
  - `profiler_backed`: `131 / 235`
  - `partial_profiler_backed`: `0`
  - `ready_missing_profiler_timing`: `0`
  - `profiler_blocked`: `6`

Target if fully closed:

- `profiler_backed`: `137 / 235`
- `profiler_blocked`: `0`

## Targets

All 6 targets are `ready` but currently `profiler_blocked`, with no complete
workload-sharded profiler evidence:

1. `L1/044_moe_expert_computation`
2. `L2/012_moe_expert_batched_execution_with_capacity_factor`
3. `L2/024_moe_expert_parallel_execution`
4. `L2/025_moe_expert_parallel_execution_backward`
5. `L2/026_moe_expert_parallel_execution_with_weighted_aggregation`
6. `L2/047_moe_training_token_repeat_and_expert_computation`

## Strategy

1. [x] Inspect existing timing sidecars and workload manifests for each target.
2. [x] Determine whether the blocker is stale import, preflight memory cap,
   rocprofv3 crash, eval_driver failure, timeout, or real reference/user OOM.
3. [x] Re-run candidates serially and workload-sharded with conservative
   settings:
   - `--max-workers 1`
   - `--no-hip-runtime-trace`
   - no fixed `--subprocess-memory-limit-gib` unless a target proves unsafe
   - one GPU device via `--gpu-device 0`
4. [ ] Import complete workload slices and aggregate problem-level sidecars.
5. [ ] Recompute coverage with new timing evidence first in the priority list.
6. [ ] Promote the result into a new canonical merged directory if coverage
   improves.

## Execution Result

Completed under the agreed validation criterion: if a problem cannot execute on
the current machine, a concrete local resource-limit conclusion counts as
completed validation.

ROCm-visible hardware:

- `rocminfo`: one `gfx1200` GPU
- VRAM pool: `16695296 KB` (`15.92 GiB`)

Existing manifests show all 6 targets were blocked by estimated timing input
footprint:

| Problem | Estimated input | Timing pool peak |
| --- | ---: | ---: |
| `L1/044_moe_expert_computation` | `21.03 GiB` | `42.05 GiB` |
| `L2/012_moe_expert_batched_execution_with_capacity_factor` | `14.11 GiB` | `28.22 GiB` |
| `L2/024_moe_expert_parallel_execution` | `12.02 GiB` | `24.04 GiB` |
| `L2/025_moe_expert_parallel_execution_backward` | `24.06 GiB` | `48.13 GiB` |
| `L2/026_moe_expert_parallel_execution_with_weighted_aggregation` | `18.06 GiB` | `36.12 GiB` |
| `L2/047_moe_training_token_repeat_and_expert_computation` | `24.03 GiB` | `48.07 GiB` |

Two least-bad candidates were re-run with elevated preflight cap, no RLIMIT_AS,
single worker, GPU 0, and no HIP runtime trace:

- `L2/024` workload offset `0`:
  `out/rdna4-close-6-profiler-blocked-20260613/probe-l2024-offset0/timing/L2/024_moe_expert_parallel_execution.timing.json`
  - result: `RUNTIME_ERROR`
  - failure class: `gen_inputs_oom_blocked`
  - generated inputs had already allocated `12.02 GiB`, then failed trying to
    allocate another `4.00 GiB` on a `15.92 GiB` GPU.
- `L2/012` workload offset `0`:
  `out/rdna4-close-6-profiler-blocked-20260613/probe-l2012-offset0/timing/L2/012_moe_expert_batched_execution_with_capacity_factor.timing.json`
  - result: `RUNTIME_ERROR`
  - failure class: `gen_inputs_oom_blocked`
  - generated inputs had already allocated `14.11 GiB`, then failed trying to
    allocate another `4.69 GiB` on a `15.92 GiB` GPU.

Conclusion:

- These 6 targets are not stale import or rocprofv3-only failures on this host.
- They need a larger-VRAM RDNA4/ROCm-visible GPU or benchmark-specific memory
  reduction that would change workload semantics.
- No new coverage was promoted because the generated evidence is negative
  `RUNTIME_ERROR` evidence, not complete profiler-backed timing.

## Safety Constraints

- Do not run multiple GPU profiling jobs concurrently.
- Do not set a low fixed RLIMIT_AS by default; L2/041 showed that this can
  create false `INVALID_REFERENCE` failures.
- Prefer workload-sharded retries and import-only aggregation over monolithic
  retries.
- Keep generated artifacts under `out/rdna4-close-6-profiler-blocked-20260613/`.

## Initial Commands

Planned primary run shape:

```bash
uv run python scripts/run_rdna4_profiler_timing_batch.py \
  --dataset-root data/SOL-ExecBench/benchmark \
  --output-dir out/rdna4-close-6-profiler-blocked-20260613/workload-sharded \
  --source-timing-dir out/rdna4-validation-reeval-20260613-latest-plus-l2041/merged \
  --only-problem-file out/rdna4-close-6-profiler-blocked-20260613/targets.txt \
  --workload-sharded \
  --timeout 1200 \
  --gpu-device 0 \
  --no-hip-runtime-trace \
  --no-resume \
  --max-workers 1
```
