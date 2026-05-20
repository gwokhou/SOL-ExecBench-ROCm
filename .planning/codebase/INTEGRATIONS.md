---
last_mapped: 2026-05-20
last_mapped_commit: unknown
focus: tech
---

# Integrations

## Dataset Sources

Benchmark data is expected under `data/`. `scripts/download_data.sh` and
`scripts/download_solexecbench.py` download datasets from Hugging Face, including
`nvidia/SOL-ExecBench` and FlashInfer trace data referenced by the README.

## GPU Runtime

The main external runtime is NVIDIA GPU infrastructure. `docker/Dockerfile`
depends on CUDA 13.1.1, cuDNN, CUTLASS, and NVIDIA driver/container support.
`scripts/run_docker.sh` launches Docker with `--gpus all`, `--ipc=host`,
`--privileged`, memory ulimits, and `FLASHINFER_TRACE_DIR`.

## PyTorch And CUDA Extensions

C++/CUDA submissions are compiled by `src/sol_execbench/driver/templates/build_ext.py`
through `torch.utils.cpp_extension.load`. The build script collects C/C++/CUDA
sources in the staging directory and emits `benchmark_kernel.so`.

Python submissions are imported directly by
`src/sol_execbench/driver/templates/eval_driver.py`. The eval driver blocks
`torch.utils.cpp_extension.load` and `load_inline` inside Python submissions so
native compilation is forced through the compile phase.

## Hardware And Clock Control

`src/sol_execbench/driver/problem_packager.py` calls `nvidia-smi` to infer local
SM versions for `LOCAL` hardware targets. Clock locking and verification live in
`src/sol_execbench/core/bench/clock_lock.py`, which uses `sudo -n nvidia-smi`
commands in the Docker environment.

## File Formats

Problem and result exchange uses JSON and JSONL:

- `definition.json` for kernel definitions.
- `workload.jsonl` for workload cases.
- `solution.json` for submission source and build metadata.
- trace JSONL emitted by `eval_driver.py` and parsed by
  `ProblemPackager.convert_stdout_to_traces`.

Safetensors inputs are supported by `src/sol_execbench/core/bench/io.py` and can
be loaded from the staging directory or `FLASHINFER_TRACE_DIR`.

## External Services

There is no application database, auth provider, web API server, or webhook
integration in the current codebase. External dependencies are package indexes,
Hugging Face dataset hosting, Docker registries, and NVIDIA GPU tooling.
