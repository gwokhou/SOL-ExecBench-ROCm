# ROCm Setup

This port targets ROCm 7.0 or newer. The current dependency lock uses PyTorch
ROCm 7.1 wheels.

## Host Requirements

- Linux host with AMD ROCm drivers and runtime installed.
- `/dev/kfd` and `/dev/dri` present and accessible.
- `rocminfo` can see at least one AMD GPU agent.
- `rocm-smi` or `amd-smi` is available for hardware inspection.
- `hipcc` is available when building HIP/C++ solutions.

Quick host checks:

```bash
rocminfo
rocm-smi
hipcc --version
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
| RDNA 4 | `gfx1200` | Full adapted suite passed locally. |
| CDNA 3 | `gfx940`, `gfx941`, `gfx942` (`gfx94*`) | Code/schema support present; hardware validation deferred. Do not claim hardware validation until a full suite run is recorded. |

The missing CDNA 3 evidence is a real `gfx94*` run of the full adapted pytest
suite, with logs and hardware/software environment details recorded in planning
artifacts before the support matrix is upgraded to hardware-validated.
