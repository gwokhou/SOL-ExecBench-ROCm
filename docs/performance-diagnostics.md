# gfx1200 performance diagnostics

Performance diagnostics are optional evidence and never change canonical Trace
timing, `T_SOL`, SOL Score, leaderboard values, or rewards.

Collect counters during evaluation with the explicit profile mode:

```bash
sol-execbench --format json evaluate PROBLEM_DIR \
  --solution SOLUTION.json \
  --profile rocprofv3-counters \
  --output TRACE.jsonl
```

The mode checks device-0 counter availability and its `gfx1200` identity with
`rocprofv3-avail`. It replays the candidate in fail-safe single-counter passes,
uses CSV as normalized input, retains ROCPD/SQLite output for audit, and hashes
the profiler, application executable, command, counter manifest, generated
configuration, and availability report. Profiler duration is not a prediction
input.

Build the governed diagnostic sidecar with:

```bash
sol-execbench --format json diagnostics performance \
  --trace TRACE.jsonl \
  --solar-analysis WORKLOAD_UUID=solar-analysis.yaml \
  --profile-summary TRACE.profile-summary.json \
  --static-evidence TRACE.static-evidence.json \
  --calibration-profile gfx1200-diagnostic-calibration.json \
  --gpu-id GPU_UUID \
  --compiler-version 'HIP version: 7.2...' \
  --power-profile stable_peak \
  --output TRACE.performance-diagnostic.json
```

Repeat `--solar-analysis` for every workload. A trusted frontier is optional and
must be supplied explicitly as
`--frontier-trace WORKLOAD_UUID=TRACE.jsonl`. Missing or mismatched GPU,
ROCm, compiler, clock, power, candidate, workload, run, hash, counter, or
cross-pass identity yields `partial`/`unavailable` with reason codes; the model
does not invent fallback evidence.

All public examples use the root `--format json` option, which must precede the
subcommand. Stdout is one versioned JSON response. Human progress messages do
not contaminate JSON-mode stdout. The primary artifact is the strict
`sol_execbench.performance_diagnostic.v1` JSON sidecar.

The host-only calibration and held-out acceptance tools are:

```bash
uv run python scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py \
  --output CALIBRATION.json --gpu-id GPU_UUID

uv run python scripts/internal/rdna4/verify_rdna4_diagnostic_acceptance.py \
  --cases HELD_OUT_CASES.json --output ACCEPTANCE.json
```

These internal tools also emit one JSON response on stdout. Calibration requires
gfx1200, HIP/ROCm 7.2-compatible tools, verified STABLE_PEAK locking, and at
least five fresh held-out process batches after tuning configuration is frozen.
