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

Alongside the normal Trace/profile/static artifacts, the command writes:

- `TRACE.jsonl.performance-timing.json`: the exact canonical trial/iteration
  samples and a deterministic 10,000-replicate hierarchical-bootstrap interval.
- `TRACE.jsonl.performance-evidence.json`: a root manifest binding definition,
  workload, solution, compile command/compiler, code objects, GPU/ROCm/clock
  identity, timing, static ISA, counter CSV, ROCPD, and counter provenance by
  SHA-256.

HIP/C++ candidates with inspectable code objects can produce complete hardware
diagnostics. Other candidate forms remain explicitly partial.

## Build the v2 diagnostic

```bash
sol-execbench --format json diagnostics performance \
  --evidence-manifest TRACE.jsonl.performance-evidence.json \
  --solar-manifest SOLAR_REQUEST/manifest.yaml \
  --calibration-profile gfx1200-diagnostic-calibration.json \
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

The output contract is `sol_execbench.performance_diagnostic.v2` using model
`gfx1200_diagnostic.v2`. It contains `T_pred(IR)`, `T_pred(HW)`, the canonical
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
request a kernel change.

## Calibration and acceptance

```bash
uv run python scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py \
  --output CALIBRATION.json --gpu-id GPU_UUID \
  --estimation-batches 5

uv run python scripts/internal/rdna4/verify_rdna4_diagnostic_acceptance.py \
  --manifest HELD_OUT_MANIFEST.json --output ACCEPTANCE.json
```

Calibration uses a frozen two-phase protocol: tuning first, then at least five
fresh parameter-estimation processes. The independent acceptance manifest must
contain at least 10 non-tuning cases for each of elementwise, transpose,
reduction/norm, and matmul (40 total). Acceptance requires median absolute
percentage error at most 15%, P90 at most 30%, and the expected primary
attribution for every family.
