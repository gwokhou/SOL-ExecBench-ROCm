# Phase 139 Verification

## Commands And Checks

Environment evidence was captured with:

```bash
UV_CACHE_DIR=/home/guohao/.cache/uv uv run sol-execbench doctor --json
rocm-smi --showproductname --showdriverversion --showclocks --showpower --showtemp --showperflevel --showclkfrq
rocprofv3 --version
rocminfo
```

Clock-lock attempt evidence was captured in:

```text
out/rdna4-timing-evidence/environment/clock-lock-attempt.json
```

Timing evidence was generated with:

```bash
UV_CACHE_DIR=/home/guohao/.cache/uv uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --phase timing --output out/rdna4-full-dataset/run --ready-subset out/rdna4-full-dataset/ready_subset.json --readiness out/rdna4-full-dataset/readiness.json --dataset-manifest out/rdna4-full-dataset/sol-dataset-manifest.json --timing-evidence-dir out/rdna4-timing-evidence/timing --gpu-architecture gfx1200 --timing-tool-version "rocprofv3 1.0.0 rocm 7.1.1" --timeout-policy record --workload-shard-size 1
```

Stability evidence was generated with `scripts/report_evaluation_stability.py`
over all 121 timing sidecars.

## Verification Results

- Environment evidence identifies RDNA4 `gfx1200`: passed.
- PyTorch ROCm device visibility is available in the `uv run` environment:
  passed.
- `rocprofv3` tool availability is recorded: passed.
- Clock-lock status is explicit: passed with blocker.
- Per-problem timing sidecars exist for all 121 ready problems: passed.
- Missing profiler-backed timing is explicit: passed; all 121 sidecars record
  fallback instead of profiler collection.
- Stability report exists and preserves non-authoritative timing boundaries:
  passed.
- Phase 138 summary was restored after the timing pass rewrote
  `out/rdna4-full-dataset/run/summary.json`: passed; it again accounts for
  121 problems, 1907 workloads, 1761 passes, and 146 failures.

## Residual Risk

- Clock lock could not complete because sudoers coverage for `rocm-smi`
  clock/reset commands appears incomplete. This keeps timing non-authoritative.
- The reference-solution timing pass selected PyTorch/event fallback policy for
  all 121 timing sidecars, so no profiler-backed `rocprofv3` kernel activity
  evidence was collected in this pass.
- Phase 140 must treat this timing set as blocker/fallback evidence, not as a
  benchmark-grade timing basis.
