---
status: complete
quick_id: 260613-close-partials-ready-missing
slug: close-partials-and-ready-missing
description: Close 3 partial profiler-backed RDNA4 targets and batch process 49 ready-missing profiler timing targets
created_at: 2026-06-13T18:45:00+08:00
---

# Quick Task 260613-close-partials-ready-missing

## Goal

Improve the latest canonical RDNA4 merged coverage by:

1. Closing the 3 remaining `partial_profiler_backed` targets.
2. Batch processing the 49 `ready_missing_profiler_timing` targets.

## Baseline

Canonical merged summary:

- `profiler_backed`: `96 / 235`
- `partial_profiler_backed`: `3`
- `ready_missing_profiler_timing`: `49`
- `reference_oom_blocked`: `46`
- `readiness_blocked`: `41`
- `profiler_blocked`: `0`

## Strategy

1. Generate target lists from canonical merged coverage.
2. Run focused workload-sharded closure for the 3 partial targets.
3. Batch ready-missing targets by category with kernel-only rocprofv3 tracing.
4. Recompute merged coverage using successful new timing outputs.
5. Summarize remaining gaps by blocker class.

## Guardrails

- Keep `--no-hip-runtime-trace` to avoid the rocprofiler-sdk HIP runtime trace
  crash path.
- Keep subprocess memory limits and dynamic preflight enabled.
- In workload-sharded mode, keep slice timing JSON compacted and aggregate from
  scalar manifest summaries; large rocprofv3 kernel traces must not be
  materialized as full `parsed_rows` lists in the parent process.
- Do not count fallback timing as profiler-backed evidence.
- Do not relax benchmark correctness semantics.

## OOM Incident

On 2026-06-13, the ready-missing serial batch was killed by the kernel OOM killer
while processing `L1/093_grouped_topk_moe_routing_backward`.

- Killed process: parent `python3 scripts/run_rdna4_profiler_timing_batch.py`
  process, PID `4141758`.
- Kernel log: anon RSS about `24.5 GiB`, total VM about `35.9 GiB`.
- No `eval_driver.py` or `rocprofv3` child process was active at the kill point.
- Immediate trigger: `L1/093` produced very large workload slice timing JSON
  files, with 16 slice sidecars totaling roughly `6.9 GiB` and the largest
  single slice sidecar about `1.36 GiB`.
- Root cause: the parent process used the default rocprofv3 timing collection
  path, which read the whole CSV, built every kernel row object, converted those
  rows into JSON dictionaries, and then wrote/reloaded large `parsed_rows`
  payloads during workload-sharded manifest aggregation.

Fix applied in code:

- `Rocprofv3CollectionRequest.compact_rows` enables streaming CSV summary mode.
- Workload-sliced `_profile_target` calls use `compact_rows=True`.
- Slice sidecars now store compacted `parsed_rows: []` plus scalar duration
  summaries; raw CSV directories are kept when `--no-compact-workload-slices`
  is requested.
- Workload manifest entries for newly completed slices are built from returned
  scalar summaries, without reloading the full sidecar JSON.
