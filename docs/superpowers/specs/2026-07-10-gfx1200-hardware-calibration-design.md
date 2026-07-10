# AMD Hardware-Model Calibration Design

## Goal

Turn AMD SOL hardware models from hand-maintained provisional roofline inputs
into reproducible, evidence-backed calibrated models. A model or individual
compute/memory path is eligible for `validated` / `supported` only when a
local HIP calibration run supplies complete, matching, stable evidence. The
packaged defaults remain provisional; calibration never silently rewrites them.

## Scope

This change covers AMD SOL hardware-model inputs and the score-authority gate
across RDNA4, CDNA3, and CDNA4 machines. It does not claim upstream NVIDIA
SOLAR equivalence, make a benchmark baseline official, or add new
operator-family estimators.

## Architecture

### Calibration workflow

A new `hardware-model calibrate` CLI command runs local HIP microbenchmarks on
one selected GPU. It detects the live gfx architecture by default; an optional
`--architecture` value is an assertion and the command rejects a mismatch. It
must run on RDNA4, CDNA3, and CDNA4 targets even where a release-quality
architecture adapter or clock-control mechanism is not yet available.

Clock control is an adapter interface. The RDNA4 adapter uses the existing
`clock_lock.lock_clocks()` and `unlock_clocks()` lifecycle. CDNA3 and CDNA4
adapters can use their own supported controls as they are added. The command
records clock state before, during, and after the benchmark and always attempts
to reset the performance policy in a `finally` path. Unsupported clock control
does not prevent diagnostic collection; it prevents an authority upgrade. The
`--require-clock-lock` option converts that evidence shortfall into a rejected
calibration for release use.

### ROCm Compute Profiler dependency environment

`rocprof-compute` is a preferred optional calibration backend. SOL ExecBench
manages a dedicated virtual environment for it under the ignored project-local
`.artifacts/rocprof-compute/<tool-version>-<requirements-sha256>/venv` path;
it never installs profiler dependencies into the project's primary `.venv` or
the system Python.

Before every invocation, the backend resolves the executable, derives the
adjacent ROCm Compute Profiler `requirements.txt`, and compares its SHA-256,
tool version, and virtual-environment Python ABI against the environment's
metadata. If the environment is absent or stale, the first invocation acquires
a per-environment file lock, creates the environment with `uv`, and installs
that exact requirements file with `uv pip`. Later calls reuse it without
manual activation or environment switching.

The subprocess runs the system ROCm executable unchanged with an isolated
environment: `PATH` starts with the managed environment's `bin`,
`VIRTUAL_ENV` names that environment, and `PYTHONNOUSERSITE=1` prevents user
site packages from contaminating the result. This works because the installed
launcher uses `env python3`; it changes only interpreter resolution, not ROCm
libraries or the executable path. The calibration artifact records the ROCm
executable path, tool version, requirements SHA-256, managed-interpreter path,
installed-distribution manifest, and backend command.

Automatic installation is enabled by default only when the user invokes the
calibration command. `--offline` prevents downloads and returns an `unknown`
backend result when dependencies are unavailable; `--no-auto-install` does the
same without attempting installation. A failed install, lock timeout, or
unreadable requirements file never falls back to ambient Python packages: the
ROCm Compute Profiler profiles are `unknown` with a stable backend reason code,
and the independent HIP probe backend may still collect its own evidence.

The command builds a dynamic capability matrix, rather than measuring one FP32
FMA value. An architecture adapter declares the candidate paths for its ISA;
the command then proves which candidates are feasible on the live host. A
candidate is feasible only if it compiles with the selected HIP toolchain,
executes on the selected GPU, and passes its minimal numerical-correctness
probe. Candidate dimensions are:

- compute operation family: vector ALU and matrix-core/instruction paths;
- input and output dtype, including each architecture's candidate FP64, FP32,
  TF32, FP16, BF16, INT8, FP8, MXFP, and successor paths;
- instruction path and layout relevant to the candidate; and
- memory access mode: read, write, and streaming copy where the adapter can
  provide a correct probe.

The initial adapters must provide portable vector and streaming-memory probes
for every supported target plus architecture-specific candidates. New ISA
paths are added declaratively to the adapter and require no CLI redesign.

When the managed ROCm Compute Profiler backend is available, calibration runs
its `--bench-only` flow and imports the raw roofline CSV rather than copying
its values into an unreferenced summary. Imported metrics are mapped to the
same capability-matrix keys and retain their raw-file checksum, profiler
version, device identity, and parser version. The HIP probe backend remains
mandatory because installed profiler versions can lack an architecture adapter
(the local ROCm 7.1.1 installation, for example, contains CDNA adapters but no
RDNA4 `gfx12*` adapter).

Each feasible candidate performs a fixed warmup and at least seven timed
samples. Its calibration value is the lowest of the best three sample values,
which is a conservative, repeatable estimate of attainable performance. A
measured candidate is accepted only if all samples are finite and positive,
there are enough valid samples, and the spread between the best and worst
retained sample is at most 5 percent. The exact iteration count, element count,
warmup count, sample durations, and statistic are saved in the artifact.

Every candidate has one of three distinct evidence states:

- `measured`: compilation, execution, numerical correctness, and stability
  checks passed; the artifact carries a calibrated throughput or bandwidth.
- `unavailable`: a probe ran and established that the current GPU, driver, or
  toolchain does not support the candidate; the artifact records that evidence
  and a stable reason code.
- `unknown`: no adapter candidate exists, a required probe could not complete,
  the toolchain/permission is unavailable, or the result is unreliable. It is
  not inferred to be unsupported and it has no default numeric value.

### Evidence and model artifacts

`sol_execbench.hardware_calibration.v1` is a JSON artifact with:

- calibration schema version, generation timestamp, command/version metadata;
- GPU UUID, architecture, card identity, ROCm version and selected device;
- each calibration backend's availability state, managed-environment
  provenance, command, raw-output checksums, and parser version;
- pre/during/post clock observations and `clock_locked` / reset outcome;
- the complete candidate capability matrix, including candidate dimensions,
  feasibility probe evidence, `measured` / `unavailable` / `unknown` state,
  raw samples, selected conservative value, spread, and reason codes;
- a distinct `collection_status` and `validation_status`, with stable blockers.

A second command, `hardware-model build`, accepts a validated calibration
artifact and writes an external `sol_execbench.amd_hardware_model.v2` JSON.
It validates the requested architecture, GPU UUID, ROCm version and artifact
schema before emitting a model. The hardware-model schema evolves from a
single `peak_tflops` / `memory_bandwidth_gbps` pair to path-keyed compute and
memory profiles. Each profile cites its calibration-artifact checksum and
path. A profile is `supported` only when its matrix item is `measured` and the
architecture adapter's validation policy, clock evidence, and environment
matching checks all pass. `unknown` and `unavailable` candidates are retained
as evidence rather than fabricated into numeric profiles.

The committed packaged models, including `data/amd_hardware_models/gfx1200.json`,
remain explicitly `inexact` / `provisional`. A release maintainer must
intentionally select the generated model through existing hardware-model
selection inputs.

### Failure behavior

The calibration command writes a rejected evidence artifact when it can gather
diagnostics but cannot establish authority. It exits nonzero for a rejected
calibration. It never produces a supported model for any of these conditions:

- the selected device cannot be identified or does not match an explicit
  architecture assertion;
- clock locking or reset is missing when `--require-clock-lock` is set;
- a candidate's HIP compilation/execution fails, values are
  non-finite/non-positive, sample count is insufficient, or retained samples
  are unstable;
- model-build input is rejected, malformed, stale relative to its supplied
  environment, or mismatches GPU UUID, architecture, or ROCm version.

Clock locking is an authority precondition, not merely a best-effort timing
optimization for any profile promoted to official-score authority. A successful
control command without an observed locked state during the benchmark is
insufficient. In diagnostic mode this condition is recorded as unverified;
the rest of the capability matrix is still collected.

### Official-score gate

`official_score_from_amd_native_score()` adds independent evidence checks in
addition to its existing numeric and baseline checks. A workload is blocked
unless all of the following are true:

- the AMD SOL aggregate bound status is `scored`;
- the SOLAR derivation aggregate status is `scored` when present;
- the hardware model resolves an exact `measured` and `supported` profile for
  the node's operation path, dtypes, and memory access; an `unknown` or
  `unavailable` profile cannot fall back to an unrelated FP32 value;
- the selected hardware profile, hardware model, and model validation state
  satisfy the architecture adapter's validated policy;
- the source score contains no `aggregate_degraded`, `aggregate_unscored`,
  `model_validation`, `hardware_validation`, `inexact_operator`, or
  `unsupported_operator` warning.

The AMD-native score artifact must carry the aggregate and hardware evidence
required by this gate. Missing evidence blocks authority; it is not interpreted
as passing. Blockers are stable, machine-readable literals and appear in the
suite summary.

## Interfaces

The following public interfaces are added:

```text
sol-execbench hardware-model calibrate \
  --device 0 --output calibration.json

sol-execbench hardware-model build \
  --calibration calibration.json --output calibrated-hardware-model.json
```

The calibration command may accept `--require-clock-lock`, `--offline`, and
`--no-auto-install`, plus bounded tuning options for tests and local diagnostics
(`--warmup-count`, `--sample-count`, candidate filtering, and problem-size
controls); release defaults are encoded in the artifact. Test-only backends
isolate HIP process execution, managed-environment creation, runtime capability
discovery, and clock adapters from artifact validation.

## Testing

Unit tests cover artifact parsing and canonical serialization; RDNA4, CDNA3,
and CDNA4 adapter candidate discovery; the three evidence states; capability
probe compilation/execution/correctness outcomes; conservative-statistic and
stability rules; clock lock/reset lifecycle including cleanup after benchmark
failures; managed ROCm Compute Profiler environment reuse, stale-environment
rebuild, environment-variable injection, offline/no-auto-install behavior, and
install failure isolation; model construction and environment mismatch
rejection; path-specific profile resolution; and official-score blockers for
every degraded, unknown, unavailable, or missing bound-evidence category.

Opt-in architecture-marked integration tests run the actual microbenchmarks on
available RDNA4, CDNA3, or CDNA4 hosts, write only under `out/`, and do not
alter packaged hardware-model data. A missing clock-control capability skips
only authority-upgrade assertions; it must not prevent basic collection tests.

## Non-goals

- Publishing a calibrated model or changing a default packaged model.
- Claiming every candidate dtype/path is available on an architecture without
  a successful probe; absent or unreliable probes remain `unknown`.
- Replacing AMD SOL operator modeling with Origami, TraceLens, or
  rocprofiler-compute.
- Generating release scoring baselines or changing suite aggregation policy.
