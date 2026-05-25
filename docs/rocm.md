# ROCm Setup

This port targets ROCm 7.0 or newer. The current dependency lock uses PyTorch
ROCm 7.1 wheels.

## Host Requirements

- Linux host with AMD ROCm drivers and runtime installed.
- `/dev/kfd` and `/dev/dri` present and accessible.
- `rocminfo` can see at least one AMD GPU agent.
- `rocm-smi` or `amd-smi` is available for hardware inspection.
- `hipcc` is available when building HIP/C++ solutions.
- ROCm library development headers are installed when running native library
  examples: `hipblas/hipblas.h`, `miopen/miopen.h`, `ck/ck.hpp`, and
  `rocwmma/rocwmma.hpp` for the supported/planned library categories.

Quick host checks:

```bash
rocminfo
rocm-smi
hipcc --version
find /opt/rocm/include -maxdepth 3 \( -path '*hipblas/hipblas.h' -o -path '*miopen/miopen.h' -o -path '*ck/ck.hpp' -o -path '*rocwmma/rocwmma.hpp' \)
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
uv run sol-execbench doctor --json
```

The JSON payload reports ROCm tool availability, best-effort GPU identity,
PyTorch ROCm runtime readiness, simple device memory behavior, and event timing
readiness. Missing tools or unavailable GPUs are diagnostic statuses, not
benchmark correctness failures.

Benchmark runs can also emit an optional environment sidecar without changing
trace JSONL:

```bash
SOLEXECBENCH_ENV_SNAPSHOT=1 \
  uv run sol-execbench examples/gemm --solution examples/gemm/solution.json \
  --json -o traces.jsonl
```

This writes `traces.jsonl.environment.json`. Use
`SOLEXECBENCH_ENV_SNAPSHOT_PATH=/path/to/env.json` to choose an explicit path.
The sidecar is reproducibility evidence only; it is not part of SOL/SOLAR
correctness or scoring.

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

The image is based on `rocm/dev-ubuntu-24.04:7.1.1-complete` and installs the
project into `/venv`.

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
passwordless `sudo`. The implementation uses `rocm-smi` to set manual
performance level and SCLK/MCLK DPM levels. If clock locking is unavailable,
run without `--lock-clocks` for functional validation.

Optional overrides:

```bash
export SOL_EXECBENCH_SCLK_LEVEL=1
export SOL_EXECBENCH_MCLK_LEVEL=1
```

## Hardware Status

Validation recorded in this milestone:

| Hardware class | Architecture | Status |
| --- | --- | --- |
| RDNA 4 | `gfx1200` | v1.8 library example scope is RDNA 4 only. |
| CDNA 3 | `gfx940`, `gfx941`, `gfx942` (`gfx94*`) | Code/schema support present; MI300X (`gfx942`) validation is prepared but deferred. Do not claim hardware validation until a full suite run and required evidence are recorded. |
| CDNA 4 | future `gfx95*` class targets | Validation deferred; not a v1.8 completion gate. |

The missing CDNA 3 evidence is a real AMD Instinct MI300X (`gfx942`) run of the
full adapted pytest suite, with logs, clock-lock evidence, dataset artifacts,
and hardware/software environment details recorded before the support matrix is
upgraded to hardware-validated. FP8 validation is expected on MI300X once
hardware access exists; NVFP4/MXFP4 validation remains deferred.

The v1.8 ROCm library ecosystem milestone uses RDNA 4 validation only. CDNA 3
and CDNA 4 library validation remain future work.
