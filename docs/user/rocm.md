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

The standalone diagnostics command does not require a problem
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
  uv run sol-execbench --format json evaluate tests/sol_execbench/samples/rmsnorm --solution tests/sol_execbench/samples/rmsnorm/solution_cuda.json \
  --trace-output traces.jsonl
```

This writes an environment-sidecar JSON file next to the chosen trace output file. Use
`SOLEXECBENCH_ENV_SNAPSHOT_PATH=out/env.json` to choose an explicit path.
The sidecar is reproducibility evidence only; it is not part of SOL/SOLAR
correctness or scoring.

## Optional Profiling Evidence

Opt-in `rocprofv3` artifact collection can diagnose anomalous or
hardware-sensitive benchmark runs:

```bash
uv run sol-execbench --format json evaluate tests/sol_execbench/samples/rmsnorm --solution tests/sol_execbench/samples/rmsnorm/solution_cuda.json \
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
| MIOpen | `miopen/miopen.h` | `-lMIOpen` | Dependency readiness is checked in the ROCm container; v3 does not ship a dedicated MIOpen sample. |
| Composable Kernel | `ck/ck.hpp` | Header-driven | Dependency readiness is checked in the ROCm container; v3 does not ship a dedicated CK sample. |
| rocWMMA | `rocwmma/rocwmma.hpp` | Header-driven | Dependency readiness is checked in the ROCm container; v3 does not ship a dedicated rocWMMA sample. |

Run `uv run pytest tests/docker/dependencies/test_rocm_libraries.py` inside the
ROCm container before evaluating a solution that uses these libraries.

## Clock Locking

`--lock-clocks` requires ROCm clock tooling to be available through
passwordless `sudo`. The implementation uses `amd-smi` to enter the firmware
`STABLE_PEAK` performance level before evaluation and returns to `AUTO` during
cleanup:

```bash
sudo amd-smi set -l STABLE_PEAK
sudo amd-smi set -l AUTO
```

Clock management records ownership. A run that changes `AUTO` to
`STABLE_PEAK` restores `AUTO` through an exception-aware context manager; a run
that finds every visible GPU already at `STABLE_PEAK` preserves that external
setting. Failed, timed-out, or interrupted lock attempts trigger a best-effort
`AUTO` rollback, and verified reset failures are reported instead of silently
discarded. Mixed or non-`AUTO` initial performance levels are not overwritten.
Owned leases that are garbage-collected without either a verified release or an
explicit ownership transfer emit a `gpu_clock_lease_leaked` error. Cross-process
Docker cleanup and deliberate `--no-reset-clocks` use explicit transfer so that
intentional retention is not reported as a leak.

The Docker image and `scripts/setup_rocm_clock_sudoers.py` install sudoers
coverage for the exact `amd-smi` commands used by the runtime. If clock locking
is unavailable, use a config with `"lock_clocks": false` only for functional
diagnostics. Such a trace is labeled with the diagnostic timing protocol and
cannot support an official claim.

The set commands apply to every visible AMD GPU because no `-g` selector is
used. The setup script is self-documenting; run it with `--help`. Its default
`check` mode is read-only, while `verify-live` explicitly exercises
`STABLE_PEAK` and guarantees an `AUTO` cleanup attempt.

## Hardware Scope

The Solution schema accepts `gfx1200`, `gfx940`, `gfx941`, `gfx942`, and
`LOCAL`. Schema acceptance is not a hardware-validation claim. Tests requiring
a real GPU declare `requires_rocm`, `requires_rdna4`, or `requires_cdna3`; their
results apply only to the device and workload surface actually executed.
