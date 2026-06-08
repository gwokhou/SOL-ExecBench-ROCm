# Phase 138 Verification

## Commands And Checks

The long-running RDNA4 run completed naturally in Codex session `9739` using:

```bash
UV_CACHE_DIR=/home/guohao/.cache/uv uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --output out/rdna4-full-dataset/run --ready-subset out/rdna4-full-dataset/ready_subset.json --readiness out/rdna4-full-dataset/readiness.json --dataset-manifest out/rdna4-full-dataset/sol-dataset-manifest.json --execution-closure out/rdna4-full-dataset/execution_closure.json --gpu-architecture gfx1200 --timeout-policy record --workload-shard-size 1 --rerun
```

Final runner output reported:

- `Total: 121 problems | OK: 86 | FAIL: 35 | SKIP: 0`
- Summary saved to `out/rdna4-full-dataset/run/summary.json`
- Per-problem traces saved under `out/rdna4-full-dataset/run`
- Execution closure saved to
  `out/rdna4-full-dataset/execution_closure.json`

Post-run artifact checks:

- `summary.json` contains 121 problem summaries.
- Per-problem `traces.json` files exist for 121 ready problems.
- The trace files contain 1895 recorded workload traces.
- `execution_closure.json` contains 3957 workload records and accounts for all
  1907 ready workloads, including the 12 missing-trace failed workloads.
- `provenance_mismatches` is empty.

## Acceptance Criteria

- Dataset denominator command succeeds: satisfied by generated dataset
  manifest, readiness, ready-subset, summary, and closure artifacts.
- Full `scripts/run_dataset.py` command completes or produces resumable partial
  artifacts with no silent omissions: satisfied; command completed naturally.
- Execution closure validates all selected workloads into explicit statuses:
  satisfied; closure records all 3957 workload rows with `attempted_passed`,
  `attempted_failed`, `missing_trace`, or `not_attempted`.
- Guardrail tests still prevent overclaiming RDNA4 dataset evidence:
  satisfied by closure claim boundary flags, all of which keep score,
  leaderboard, paper parity, and full-validation authority false.

## Residual Risk

- The run was not clock-locked and did not collect `rocprofv3` profiler
  evidence in this phase. That is intentionally deferred to Phase 139.
- Failed workloads are real RDNA4 findings, not accepted passes. They must
  remain visible in Phase 140 reports and Phase 141 public wording.
- `hf download flashinfer-ai/flashinfer-trace` remained a separate background
  task and was not required for this ready-subset execution because FlashInfer
  workloads were readiness-blocked.
