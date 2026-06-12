---
status: complete
created_at: 2026-06-13T00:00:00+08:00
task: rdna4-validation-100-percent
---

# RDNA4 Validation 100 Percent Push Plan

Goal: use the latest RDNA4 v1.35 validation completion reports to push the
remaining validation work toward 100% without overstating claim boundaries.

## Checks

- [x] Recompute profiler timing coverage from the latest v1.35 timing evidence.
- [x] Classify which remaining rows are runnable evidence gaps versus current
      host/readiness blockers.
- [x] Remove the batch-script gap that prevented ready-but-missing-profiler rows
      from being selected for follow-up profiling.
- [x] Verify target selection and focused regression tests.
- [x] Record the remaining strict-coverage blockers.

## Findings

Latest v1.35 coverage remains:

- denominator: 235 problems;
- full profiler-backed: 88 problems;
- partial profiler-backed: 28 problems;
- ready missing profiler timing: 73 problems;
- reference OOM-blocked: 5 problems;
- readiness-blocked: 41 problems;
- full profiler-backed timing coverage: false.

The original profiler batch target selector only selected
`timing_fallback`, `partial_profiler_backed`, and `profiler_blocked`. Since the
v1.35 report has 0 `timing_fallback` and 0 `profiler_blocked`, it could not
advance the 73 `ready_missing_profiler_timing` rows even though those are the
next profiler-runnable gap.

## Result

Added explicit target-status selection to
`scripts/run_rdna4_profiler_timing_batch.py`. The default target statuses remain
unchanged, while operators can now run:

```bash
uv run python scripts/run_rdna4_profiler_timing_batch.py \
  --source-timing-dir out/rdna4-v135-rerun-20260611/profiler-batch/timing \
  --source-timing-dir out/rdna4-profiler-sharded-closure-l1026-20260608/timing \
  --source-timing-dir out/rdna4-profiler-workload-aggregate-20260608-v2/timing \
  --source-timing-dir out/rdna4-profiler-backed-timing-full-20260608/timing \
  --source-timing-dir out/rdna4-timing-evidence/timing \
  --target-status ready_missing_profiler_timing \
  --workload-sharded \
  --strict-isolation \
  --gpu-device 0
```

Pure selection verification confirmed that the new status selector targets all
73 ready-missing problems covering 1244 workloads.

Strict 100% full profiler-backed timing coverage is not claimable from the
current evidence because at least 41 readiness-blocked and 5 reference
OOM-blocked denominator rows remain non-profiler-backed on the recorded 16GB
RDNA4 host.
