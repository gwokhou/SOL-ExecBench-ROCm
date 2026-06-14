---
status: in_progress
quick_id: 260613-iut
slug: close-rdna4-ready-missing-profiler-timin
description: Close RDNA4 ready_missing_profiler_timing gap for 73 ready problems
created_at: 2026-06-13T05:34:34.075Z
---

# Quick Task 260613-iut: Close RDNA4 Ready-Missing Profiler Timing Gap

## Goal

Close the `ready_missing_profiler_timing` gap for the 73 RDNA4-ready problems
reported by the merged 2026-06-13 evaluation artifacts.

The success condition is to convert as many of these 73 problems as possible
from `ready_missing_profiler_timing` to `profiler_backed`, with explicit
blocker evidence for every target that cannot be converted on the current
RDNA4 validation host.

## Baseline

- Merged evaluation root:
  `out/rdna4-validation-reeval-20260613-latest/merged/`
- Baseline coverage:
  `out/rdna4-validation-reeval-20260613-latest/merged/profiler-timing-coverage-summary.json`
- Baseline status:
  - `profiler_backed_problems`: 88 / 235
  - `profiler_backed_coverage_pct`: 37.4468
  - `ready_missing_profiler_timing_problems`: 73
  - `profiler_blocked_problems`: 0
  - `partial_profiler_backed_problems`: 2

## Target Set

Target list:
`ready-missing-profiler-targets.txt`

Target count: 73 problems.

Category split:

| Category | Problems |
| --- | ---: |
| FlashInfer-Bench | 17 |
| L1 | 22 |
| L2 | 33 |
| Quant | 1 |

## Execution Plan

1. Generate a fresh baseline audit from the merged artifacts before running any
   new profiler batch.
2. Run `scripts/run_rdna4_profiler_timing_batch.py` against
   `ready-missing-profiler-targets.txt` with strict GPU isolation and workload
   sharding enabled where needed.
3. Split execution into conservative category or workload shards so failures in
   long-tail L2 targets do not block easier L1, FlashInfer reference-compatible,
   or Quant targets.
4. Recompute profiler timing coverage using the new timing output first,
   followed by existing evidence directories from the merged evaluation.
5. Rebuild the sharded closure audit and blocker ledger.
6. Update the merged evaluation artifacts or create a new merged evaluation root
   with the new coverage, closure audit, summary, and explicit residual gap
   accounting.

## Candidate Commands

Baseline target extraction is already recorded in
`ready-missing-profiler-targets.txt`.

Use a new output root such as:

```bash
out/rdna4-ready-missing-profiler-closure-20260613
```

Run profiler batches with the repository script. Exact flags should follow the
current `run_rdna4_profiler_timing_batch.py` interface after checking `--help`,
but the run should include:

- `--dataset-root data/SOL-ExecBench/benchmark`
- repeated `--only-problem <problem_id>` arguments generated from
  `.planning/quick/260613-iut-close-rdna4-ready-missing-profiler-timin/ready-missing-profiler-targets.txt`
- `--workload-sharded`
- `--strict-isolation`
- `--gpu-device 0`
- output under `out/rdna4-ready-missing-profiler-closure-20260613/`

The script does not currently accept a problem-list file directly. Build the
`--only-problem` argument list from `ready-missing-profiler-targets.txt`, or run
per-category/per-shard batches by passing repeated `--only-problem` flags.

Coverage recomputation should include the new timing directory before existing
evidence directories:

```bash
uv run python scripts/run_rdna4_profiler_timing_coverage.py \
  --dataset-root data/SOL-ExecBench/benchmark \
  --output-dir out/rdna4-ready-missing-profiler-closure-20260613/coverage \
  --timing-evidence-dir out/rdna4-ready-missing-profiler-closure-20260613/profiler-batch/timing \
  --timing-evidence-dir out/rdna4-short-term-closure-20260613/profiler-batch-final-two/timing \
  --timing-evidence-dir out/rdna4-short-term-closure-20260613/profiler-batch/timing \
  --timing-evidence-dir out/rdna4-v135-rerun-20260611/profiler-batch/timing \
  --timing-evidence-dir out/rdna4-profiler-workload-aggregate-20260608-v2/timing \
  --timing-evidence-dir out/rdna4-profiler-backed-timing-full-20260608/timing \
  --timing-evidence-dir out/rdna4-timing-evidence/timing
```

## Verification

- `scripts/run_rdna4_profiler_timing_batch.py` completes or records explicit
  per-target blocker evidence.
- `scripts/run_rdna4_profiler_timing_coverage.py` exits 0.
- `scripts/run_rdna4_profiler_sharded_closure.py` exits 0.
- Final coverage summary reports:
  - updated `profiler_backed_problems`
  - updated `ready_missing_profiler_timing_problems`
  - updated `partial_profiler_backed_problems`
  - updated blocker class counts
- Every original target from `ready-missing-profiler-targets.txt` is accounted
  for as one of:
  - converted to `profiler_backed`
  - converted to `partial_profiler_backed`
  - explicitly blocker-classified
  - still `ready_missing_profiler_timing` with a recorded reason

## Out Of Scope

- Closing the two existing partial profiler-backed targets from the separate
  short-term closure task.
- Porting readiness-blocked Quant CUDA paths.
- Reclassifying FlashInfer runtime-blocked problems.
- Solving current-device OOM denominator policy.
- Claiming score authority, paper parity, or leaderboard readiness.

## Execution Notes

### 2026-06-13 Stage 1

Output root:
`out/rdna4-ready-missing-profiler-closure-20260613/`

Implementation adjustment:

- Updated `scripts/run_rdna4_profiler_timing_batch.py` target selection so
  `ready_missing_profiler_timing` problems are eligible batch targets.
- Added regression coverage in
  `tests/sol_execbench/test_rdna4_profiler_timing_batch.py`.
- Verification: `uv run pytest tests/sol_execbench/test_rdna4_profiler_timing_batch.py`
  passed with 34 tests.

Profiler batch results:

- FlashInfer-Bench: 17/17 converted to profiler-backed timing evidence.
- L1: 21/22 converted in `l1-batch`; the final target
  `L1/094_time_decay_exponential_stabilization` was retried with workspace
  `TMPDIR` and remains `profiler_blocked`.
- Quant: `Quant/014_fp8_yarn_rope_embedding` remains `profiler_blocked`.
- L2 chunk 1:
  - `L2/006_multimodal_rope_position_calculation`
  - `L2/009_decoder_layer_with_residual_connections`
  - `L2/010_moe_expert_computation_with_weighted_accumulation`
  - `L2/011_moe_sparse_routing_and_dispatch_backward`
  all produced timing files but remain `profiler_blocked` because rocprofv3
  evidence was not accepted as full profiler-backed coverage.

Operational notes:

- `/tmp` filled during the first L1 run. Subsequent runs used workspace
  `TMPDIR=out/rdna4-ready-missing-profiler-closure-20260613/tmp` and
  `UV_CACHE_DIR=out/rdna4-ready-missing-profiler-closure-20260613/uv-cache`.
- The first L2 chunk took roughly 34 minutes. The auto-chained L2 runner was
  interrupted after chunk 1 so remaining L2 targets can be handled with a more
  controlled single-target or smaller-shard strategy.

Stage 1 coverage recomputation:
`out/rdna4-ready-missing-profiler-closure-20260613/coverage-stage1/`

- `profiler_backed_problems`: 121 / 235
- `profiler_backed_coverage_pct`: 51.4894
- `ready_missing_profiler_timing_problems`: 29
- `profiler_blocked_problems`: 6
- `partial_profiler_backed_problems`: 3

Residual ready-missing targets are all L2 problems. Profiler-blocked targets
are:

- `L1/094_time_decay_exponential_stabilization`
- `L2/006_multimodal_rope_position_calculation`
- `L2/009_decoder_layer_with_residual_connections`
- `L2/010_moe_expert_computation_with_weighted_accumulation`
- `L2/011_moe_sparse_routing_and_dispatch_backward`
- `Quant/014_fp8_yarn_rope_embedding`
