# ROCm Setup

This port targets ROCm 7.0 or newer. The current dependency lock uses PyTorch
ROCm 7.2 wheels.

## Host Requirements

- Linux host with AMD ROCm drivers and runtime installed.
- `/dev/kfd` and `/dev/dri` present and accessible.
- `rocminfo` can see at least one AMD GPU agent.
- `amd-smi` is available for hardware inspection and clock-state management.
- `hipcc` is available when building HIP/C++ solutions.
- ROCm library development headers are installed when running native library
  examples: `hipblas/hipblas.h`, `miopen/miopen.h`, `ck/ck.hpp`, and
  `rocwmma/rocwmma.hpp` for the supported/planned library categories.

Quick host checks:

```bash
rocminfo
amd-smi list
amd-smi metric -l
hipcc --version
find /opt/rocm/include -maxdepth 3 \
  \( -path '*hipblas/hipblas.h' \
  -o -path '*miopen/miopen.h' \
  -o -path '*ck/ck.hpp' \
  -o -path '*rocwmma/rocwmma.hpp' \)
```

## Python Environment

Install dependencies with:

```bash
uv sync --all-groups
```

The Linux dependency source is the PyTorch ROCm index configured in
`pyproject.toml`. Verify that CUDA wheels were not installed:

```bash
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
print(torch.cuda.get_device_properties(0).name)
print(torch.cuda.get_device_properties(0).gcnArchName)
PY
```

Expected result: `torch.version.hip` is set, `torch.version.cuda` is `None`,
and the device reports an AMD architecture such as `gfx1200`.

## Runtime Diagnostics

v1.13 provides a standalone diagnostics command that does not require a problem
directory or solution:

```bash
uv run sol-execbench --format json environment doctor
```

The JSON payload reports ROCm tool availability, best-effort GPU identity,
PyTorch ROCm runtime readiness, simple device memory behavior, and event timing
readiness. Missing tools or unavailable GPUs are diagnostic statuses, not
benchmark correctness failures.

Benchmark runs can also emit an optional environment sidecar without changing
trace JSONL:

```bash
SOLEXECBENCH_ENV_SNAPSHOT=1 \
  uv run sol-execbench --format json evaluate examples/hip_cpp/rmsnorm --solution examples/hip_cpp/rmsnorm/solution_hip.json \
  --trace-output traces.jsonl
```

This writes an environment-sidecar JSON file next to the chosen trace output file. Use
`SOLEXECBENCH_ENV_SNAPSHOT_PATH=out/env.json` to choose an explicit path.
The sidecar is reproducibility evidence only; it is not part of SOL/SOLAR
correctness or scoring.

## Optional Profiling Evidence

v1.14 adds opt-in `rocprofv3` artifact collection for diagnosing anomalous or
hardware-sensitive benchmark runs:

```bash
uv run sol-execbench --format json evaluate examples/hip_cpp/rmsnorm --solution examples/hip_cpp/rmsnorm/solution_hip.json \
  --profile rocprofv3 --trace-output traces.jsonl
```

Profiling is disabled by default. When enabled, SOL ExecBench writes profiler
artifacts under `traces.jsonl.rocprofv3/` and writes diagnostic metadata JSON beside
the output artifact. The metadata records the profiler command,
working directory, timeout, output locations, registered artifacts, exit status,
and stdout/stderr tails.

The profiling sidecar is optional evidence only. It does not add fields to
canonical trace JSONL, does not change correctness status, and is not used as
score authority. If `rocprofv3` is missing or the profiler command fails, the
CLI records an unavailable or failed status and then runs the normal benchmark
path.

`rocprofv3` usually needs the same ROCm device access as benchmark execution:
`/dev/kfd`, `/dev/dri`, the `video` group, and compatible host/container ROCm
tools. In Docker, use the standard `./scripts/run_docker.sh` wrapper so those
devices and security options are present.

## Docker

Build and enter the ROCm container:

```bash
./scripts/run_docker.sh --build
```

The wrapper refuses Docker Desktop contexts because ROCm device passthrough
requires the native Linux Docker daemon. It runs containers with:

- `--device=/dev/kfd`
- `--device=/dev/dri`
- `--group-add video`
- `--security-opt seccomp=unconfined`
- `--ipc=host`

The default declared target is based on
`rocm/dev-ubuntu-24.04:7.2-complete` and builds the local image as
`sol-execbench:rocm-7.2-complete`. The image installs the project into
`/venv`. Non-default declared targets keep the same wrapper flow but pass
target-specific Docker build arguments for the PyTorch ROCm wheel stack, such
as `torch==2.10.0+rocm7.0` for ROCm 7.0 and `torch==2.11.0+rocm7.2` for ROCm
7.2. Targets that are still classified as `not_tested` are blocked by default;
use `--allow-untested-target-smoke` only for smoke/E2E diagnostics that should
not be treated as validated Matrix evidence.

## ROCm Library Dependencies

Runnable native library examples rely on ROCm development packages in addition
to HIP itself:

| Library category | Headers | Link/runtime library | Notes |
| --- | --- | --- | --- |
| hipBLAS | `hipblas/hipblas.h` | `-lhipblas` | Supported by the existing SGEMM example. |
| MIOpen | `miopen/miopen.h` | `-lMIOpen` | Supported by `examples/miopen/softmax/` for softmax/cuDNN-style coverage. |
| Composable Kernel | `ck/ck.hpp` | Header-driven for the example path | Supported by `examples/ck/gemm/` for a small GEMM coverage path. |
| rocWMMA | `rocwmma/rocwmma.hpp` | Header-driven for the example path | Supported by `examples/rocwmma/gemm/` for RDNA 4 matrix-core GEMM coverage. |

Run `uv run pytest tests/docker/dependencies/test_rocm_libraries.py` inside the
ROCm container to check these dependencies before attempting the library
examples.

## Clock Locking

`--lock-clocks` requires ROCm clock tooling to be available through
passwordless `sudo`. The implementation uses `amd-smi` to enter the firmware
`STABLE_PEAK` performance level before evaluation and returns to `AUTO` during
cleanup:

```bash
sudo amd-smi set -l STABLE_PEAK
sudo amd-smi set -l AUTO
```

The Docker image and `scripts/setup_rocm_clock_sudoers.py` install sudoers
coverage for the exact `amd-smi` commands used by the runtime. If clock locking
is unavailable, run without `--lock-clocks` for functional validation.

The set commands apply to every visible AMD GPU because no `-g` selector is
used. The setup script is self-documenting; run it with `--help`. Its default
`check` mode is read-only, while `verify-live` explicitly exercises
`STABLE_PEAK` and guarantees an `AUTO` cleanup attempt.

## Engineering Prerelease Support Matrix

The v1.25 engineering prerelease separates what has evidence from what is
deferred or unavailable. These rows are support-boundary statements, not paper
parity, upstream SOLAR parity, hosted leaderboard readiness, hard-sandbox
authority, or new hardware-validation claims.
Use engineering-prerelease evidence wording for these rows so container,
RDNA 4, CDNA3, and CDNA4 statements stay bounded to the artifacts named in the
project docs.

| Support surface | Architecture or scope | Engineering prerelease status | Interpretation |
| --- | --- | --- | --- |
| RDNA 4 | `gfx1200` | Historical v1.35 same-source rerun recorded `derived_evidence_missing=0`, 121 replacement timing sidecars/workload manifests, 88 full profiler-backed problems, 28 partial profiler-backed problems, 0 fallback/profiler-blocked problems, 73 ready-missing timing problems, 0 rebuilt consistency findings, and `overall_status=passed`. The ignored raw output tree has been pruned. | Valid bounded RDNA4 evidence for the recorded host and historical artifacts; full profiler-backed timing coverage remains false, timing remains non-authoritative, and the result is not full 235-problem paper validation, upstream SOLAR parity, NVIDIA B200 equivalence, hosted leaderboard authority, CDNA3/MI300X validation, CDNA4 validation, or broader AMD hardware validation. |
| Docker/container ROCm user-space | Declared ROCm container targets on recorded host driver/devices | Container user-space evidence can be recorded for selected ROCm targets. | Docker/container ROCm user-space evidence is not native-host validation and must not be used as native-host, score, paper-parity, or leaderboard authority. |
| MI300X GPU under CDNA 3 | MI300X and MI308X are sibling GPU products under the CDNA3 architecture family and share the `gfx942` code path; `gfx940`, `gfx941`, and `gfx942` remain CDNA 3 code/schema targets. | CDNA3 validation infrastructure evidence exists on MI308X (`gfx942`): pytest passed, the full dataset run executed, Quant NVFP4/MXFP4 skips were expected, and remaining non-skip blockers are timeout shards. | This is not a completed benchmark-grade MI300X hardware-validation claim because MI308X and MI300X hardware configurations differ, despite sharing `gfx942`. |
| CDNA4 | Future CDNA4 class hardware | CDNA4 validation is unavailable because suitable hardware is not currently accessible. | CDNA4 is not a v1.25 validation target and should be reported as unavailable, not as validated or merely skipped. |

NVFP4/MXFP4 Quant benchmark ROCm adaptation is deliberately deferred while no
CDNA4-class hardware is available. The project does not replace CUDA-only
scaled-matrix-multiply reference semantics with a dequantized fallback for
benchmark validation; CDNA3 runs record these problems as expected
`cdna3_low_precision_hardware_unsupported` skips instead.

## Hardware Status

Validation recorded in this milestone:

| Hardware class | Architecture | Status |
| --- | --- | --- |
| RDNA 4 | `gfx1200` | v1.35 same-source rerun evidence exists with `derived_evidence_missing=0`, 121 replacement profiler timing sidecars, 88 full profiler-backed problems, 28 partial profiler-backed problems, 0 rebuilt consistency findings, a passing prerelease artifact bundle, and an explicit `full_profiler_backed_timing_coverage=false` boundary. |
| CDNA 3 | `gfx940`, `gfx941`, `gfx942` (`gfx94*`) | Code/schema support present. Real MI308X (`gfx942`) runs recorded a passed adapted pytest suite and an operational full-dataset validation path with known timeout blockers. Do not claim full MI300X hardware validation until those blockers and required exact-hardware MI300X evidence are resolved or explicitly bounded. |
| CDNA 4 | future CDNA4 class targets | CDNA4 validation is unavailable because suitable hardware is not currently accessible. |

The current CDNA3 evidence is bounded. A real MI308X (`gfx942`) cloud run
passed the adapted pytest suite, and a later 235-problem dataset validation run
showed 220 complete passing problem traces, 15 expected Quant NVFP4/MXFP4 skips
on CDNA3, and 6 timeout shards across 4 problems. The nested timeout
classification path was fixed and targeted verification confirmed that timed
out shards are now recorded as `TIMEOUT` traces and summary failures.

The missing benchmark-grade MI300X evidence is an exact MI300X environment
record plus a timeout-free or explicitly bounded dataset result, clock-lock
evidence, timing artifacts, AMD-native score artifacts, and FP8 status. MI308X
and MI300X share the `gfx942` code path but have different hardware
configurations, so MI308X evidence cannot complete MI300X validation. Timing
from the tested cloud environment must remain unlocked-clock evidence until the
host can report locked clocks. NVFP4/MXFP4 validation remains deferred on CDNA3
and requires CDNA4-class hardware support.

CDNA3 test readiness is now concrete but still bounded: the repository contains
`tests/sol_execbench/core/platform/test_cdna3_hardware_marker.py`, which is selected with
`requires_cdna3` and runs only on `gfx94*` ROCm targets. The status may change
from infrastructure evidence to MI300X hardware-validated only after the MI300X
handoff artifacts exist for actual MI300X hardware and
`mi300x_validation_claim_blockers()` returns no blockers.

The v1.8 ROCm library ecosystem milestone uses RDNA 4 validation only.
MI300X library validation remains future work, and CDNA4 validation is
unavailable until suitable hardware is accessible.
