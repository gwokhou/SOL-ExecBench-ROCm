# gfx1200 performance diagnostics

Performance diagnostics are diagnostic-only. They never change canonical Trace
timing, `T_SOL`, SOL Score, leaderboard values, or rewards.

## Collect one governed evidence bundle

Counter mode accepts exactly one workload and requires persisted static
evidence:

```bash
sol-execbench --format json evaluate PROBLEM_DIR \
  --solution SOLUTION.json \
  --workload-uuid WORKLOAD_UUID \
  --profile rocprofv3-counters \
  --static-evidence auto \
  --output TRACE.jsonl
```

The unprofiled canonical run executes first. Only after it succeeds does
rocprofv3 replay that workload in fail-safe counter passes. Replay stdout and
timing never become canonical timing. Overlapping/multi-queue dispatches are
unsupported and fail closed.

Every selected counter group must first pass an exact
`rocprofv3-avail -d 0 pmc-check ...` invocation. A rejected or timed-out group
fails closed before replay. The combined command/output digest is stored in
the counter provenance alongside availability, profiler, configuration, and
application hashes.

Alongside the normal Trace/profile/static artifacts, the command writes:

- `TRACE.jsonl.performance-timing.json`: the exact canonical trial/iteration
  samples and a deterministic 10,000-replicate hierarchical-bootstrap interval.
- `TRACE.jsonl.performance-replay.json`: exact input hash, process executable,
  10-warmup/5-evidence ROCTx markers, cache policy, pre/post AMD SMI telemetry,
  and cross-pass dispatch sequence identity.
- `TRACE.jsonl.performance-evidence.json`: a root manifest binding definition,
  workload, solution, compile command/compiler, code objects, GPU/ROCm/clock
  identity, timing, static ISA, counter CSV, ROCPD, and counter provenance by
  SHA-256.

HIP/C++ candidates with inspectable code objects can produce complete hardware
diagnostics. Other candidate forms remain explicitly partial.

ROCm 7.2 containers set `ROCPROF_TMPDIR=/tmp`. This avoids the upstream
rocprofv3 ring-buffer failure caused by constructing profiler temporary paths
from an unsuitable container working directory.

## Freeze inference and build the v3 diagnostic

Development and held-out corpora contain only labels and content-addressed
evidence/SOLAR references. They cannot contain supplied predictions. Each
corpus contains at least 20 cases from each supported family (80 total), and
their workload/candidate pair IDs must be disjoint.

The current corpus contract derives each pair ID from the evidence-bound
workload SHA-256 and candidate SHA-256. Authoring re-derives that identity,
checks the declared family against the built diagnostic, and rejects reused
evidence manifests across development and held-out data. A caller-provided
`independent=true` assertion is not accepted.

```bash
sol-execbench --format json diagnostics fit-performance-inference \
  --development-corpus DEVELOPMENT.json \
  --calibration-profile gfx1200-diagnostic-calibration.json \
  --output gfx1200-diagnostic-inference.json
```

This freezes family-specific 95% split-conformal expansion factors and
deterministic action thresholds before held-out data is read.

```bash
sol-execbench --format json diagnostics performance \
  --evidence-manifest TRACE.jsonl.performance-evidence.json \
  --solar-manifest SOLAR_REQUEST/manifest.yaml \
  --calibration-profile gfx1200-diagnostic-calibration.json \
  --inference-profile gfx1200-diagnostic-inference.json \
  --output TRACE.performance-diagnostic.json
```

A trusted frontier is optional:

```text
--frontier-trace FRONTIER.jsonl
```

The command only consumes manifest-bound identities; GPU/compiler/power
identity cannot be supplied manually. The SOLAR manifest must cite an eligible
analysis for the exact `definition:workload_uuid`. The current narrow admission
set is:

- contiguous FP32/BF16 elementwise graphs;
- 2D contiguous, out-of-place FP16/BF16/FP32 transpose;
- last-axis sum/mean/RMSNorm with BF16/FP32 input and FP32 accumulation;
- contiguous FP16 GEMM/BMM with FP32 accumulation and output.

Unsupported semantics, missing fusion regions, hash or identity mismatch,
counter-pass misalignment, missing queue identity, or overlap produces
`partial`/`unavailable` reason codes. No representative shape, achieved-rate,
profiler-duration, or measured-runtime fallback is permitted.

The output contract is `sol_execbench.performance_diagnostic.v3` using model
`gfx1200_diagnostic.v3`. It contains `T_pred(IR)`, `T_pred(HW)`, the canonical
measured confidence interval, optional trusted frontier, uncertainty-aware
`L/C/R`, bounded attribution, and stable action codes.

## Govern Agent feedback

Code-changing recommendations require a current diagnostic and an accepted
held-out model report:

```bash
sol-execbench --format json diagnostics agent-feedback \
  --performance-diagnostic TRACE.performance-diagnostic.json \
  --evidence-manifest TRACE.jsonl.performance-evidence.json \
  --acceptance ACCEPTANCE.json \
  --output TRACE.performance-agent-feedback.json
```

Partial diagnostics may only produce reprofile/model-gap actions. They cannot
request a kernel change. If `--acceptance` is omitted, or a matching report
records a failed verdict, the output is still generated but contains only
those safe actions. Identity or hash mismatch is an input error.

## Calibration and acceptance

```bash
uv run python scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py \
  --output CALIBRATION.json --gpu-id GPU_UUID \
  --estimation-batches 5

sol-execbench --format json diagnostics accept-performance-model \
  --development-corpus DEVELOPMENT.json \
  --held-out-corpus HELD_OUT.json \
  --calibration-profile CALIBRATION.json \
  --inference-profile INFERENCE.json \
  --manifest-output ACCEPTANCE-MANIFEST.json \
  --output ACCEPTANCE.json
```

Calibration uses a frozen two-phase protocol: tuning first, then at least five
fresh parameter-estimation processes. Calibration and replay audits include
stable pre/post GPU identity, clock, temperature, power, and foreign-process
observations. Acceptance requires at least 20 held-out cases per family (80
total), at least 90% empirical interval coverage in every family, median
absolute percentage error at most 15%, P90 at most 30%, and at least 90%
precision plus 70% recall for every enabled code-changing action.

## Four-family hardware smoke

Before collecting the full validation corpora, verify one complete,
content-addressed case from each supported family. Set
`SOL_EXECBENCH_DIAGNOSTIC_SMOKE_JSON` to a root-confined configuration:

```json
{
  "schema_version": "diagnostic_smoke_test.v1",
  "calibration_profile": "calibration/gfx1200-diagnostic-v3.json",
  "cases": [
    {
      "workload_kind": "elementwise",
      "evidence_manifest": "elementwise/performance-evidence.json",
      "solar_manifest": "elementwise/solar/manifest.yaml"
    },
    {
      "workload_kind": "transpose",
      "evidence_manifest": "transpose/performance-evidence.json",
      "solar_manifest": "transpose/solar/manifest.yaml"
    },
    {
      "workload_kind": "reduction_norm",
      "evidence_manifest": "reduction/performance-evidence.json",
      "solar_manifest": "reduction/solar/manifest.yaml"
    },
    {
      "workload_kind": "matmul",
      "evidence_manifest": "matmul/performance-evidence.json",
      "solar_manifest": "matmul/solar/manifest.yaml"
    }
  ]
}
```

Then run:

```bash
uv run pytest \
  tests/sol_execbench/core/bench/test_rdna4_performance_diagnostics_smoke.py
```

The smoke requires available IR and HW predictions plus available `C` and `R`
for all four cases. A missing configuration skips the optional hardware test
and is not evidence that the smoke passed.
