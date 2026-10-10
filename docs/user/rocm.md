# ROCm Setup

This port supports Linux x86-64 with Python 3.14. The current dependency lock
uses ROCm 10.1.0 and AMD's PyTorch 2.14.0 wheels.

## Host Requirements

- Linux x86-64 host with compatible AMD ROCm drivers installed.
- `/dev/kfd` and `/dev/dri` present and accessible.
- `rocminfo` can see at least one AMD GPU agent.
- `amd-smi` is available for hardware inspection and clock-state management.
- `hipcc` is available when building HIP/C++ solutions.
- ROCm library development headers are installed when running native library
  examples: `hipblas/hipblas.h`, `miopen/miopen.h`, `ck/ck.hpp`, and
  `rocwmma/rocwmma.hpp` for the supported library categories.

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
uv sync --locked --all-groups
uv run rocm-sdk init
export ROCM_PATH="$(uv run rocm-sdk path --root)"
export HIP_PATH="${ROCM_PATH}"
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"
```

The dependency source is AMD's ROCm 10.1 wheel index configured in
`pyproject.toml`. The exported paths select the wheel SDK for native tools.
Verify the runtime:

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
- `--network=none` and `--cap-drop=ALL`
- the Docker default seccomp policy plus
  `--security-opt no-new-privileges=true`
- private IPC with an 8 GiB shared-memory allocation
- a 512-process/container PID ceiling

The only declared build target is Linux x86-64 with Python 3.14, based on
`rocm/pytorch:rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0`.
The Dockerfile pins the official manifest digest and builds the local image as
`sol-execbench:rocm-10.1.0`. The project lives in `/venv`, which inherits the
base image's PyTorch, torchvision, Triton, and ROCm SDK through system
site-packages. Additional project dependencies are installed from `uv.lock`;
the GPU stack is checked against the declared target and is never reinstalled.
The wheel SDK development root is exposed at `/opt/rocm` for native tools.
PyTorch-bound HIP/C++ extensions require C++20; packaged templates and native
samples use `-std=c++20`.
Other operating systems, architectures, and former ROCm 7.x build targets are
unsupported. Targets classified as `not_tested` are blocked by default; use
`--allow-untested-target-smoke` only for smoke/E2E diagnostics that should not
be treated as validated Matrix evidence.

## ROCm Library Dependencies

Runnable native library examples rely on ROCm development packages in addition
to HIP itself:

| Library category | Headers | Link/runtime library | Notes |
| --- | --- | --- | --- |
| hipBLAS | `hipblas/hipblas.h` | `-lhipblas` | Supported by the existing SGEMM example. |
| MIOpen | `miopen/miopen.h` | `-lMIOpen` | Focused solution metadata is checked in under `tests/sol_execbench/samples/rdna4_vecadd/`. |
| Composable Kernel | `ck/ck.hpp` | Header-driven | Focused solution metadata is checked in under `tests/sol_execbench/samples/rdna4_vecadd/`. |
| rocWMMA | `rocwmma/rocwmma.hpp` | Header-driven | A focused RDNA4 sample is checked in under `tests/sol_execbench/samples/rdna4_rocwmma/`. |

The pinned official ROCm 10.1 image does not provide Composable Kernel headers.
CK samples require a separate compatible header installation; their tests use
`requires_ck` and skip only when those headers are absent.

Run `uv run pytest tests/docker/dependencies/test_rocm_libraries.py -m
docker_dependency -n 0` inside the ROCm container before evaluating a solution
that uses these libraries.

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
