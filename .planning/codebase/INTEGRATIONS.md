---
generated_by: gsd-map-codebase
focus: tech
mapped_at: 2026-06-16
---

# Integrations

## Runtime Integrations

- PyTorch ROCm is the primary execution integration. It is imported by the
  evaluation driver in `src/sol_execbench/driver/templates/eval_driver.py` and
  used by timing, correctness, input generation, and native extension loading.
- Triton ROCm is a supported solution category through `triton-rocm==3.6.0` and
  example problems under `examples/triton/`.
- HIP/C++ native extension compilation is routed through
  `torch.utils.cpp_extension` in `src/sol_execbench/driver/templates/build_ext.py`.
- ROCm command-line tools are probed by helpers such as
  `src/sol_execbench/driver/problem_packager.py`,
  `src/sol_execbench/core/bench/clock_lock.py`, and
  `src/sol_execbench/core/diagnostics.py`.

## ROCm Evidence Tools

- `rocprofv3` diagnostic profiling is represented by
  `src/sol_execbench/core/bench/rocm_profiler.py`.
- Static kernel evidence is collected through bounded external tool calls in
  `src/sol_execbench/core/bench/static_kernel_evidence.py`.
- Toolchain routing is modeled in `src/sol_execbench/core/toolchain.py` and
  exposed through `sol-execbench toolchain --json`.
- Environment diagnostics and snapshots are modeled in
  `src/sol_execbench/core/environment.py` and surfaced through
  `sol-execbench doctor --json`.

## Docker Integration

- `scripts/run_docker.sh` is the operator wrapper for ROCm container runs.
- `docker/Dockerfile` builds the image with ROCm user-space paths such as
  `ROCM_PATH=/opt/rocm`, `HIP_PATH=/opt/rocm`, and `HIP_PLATFORM=amd`.
- `docker/entrypoint.sh` performs runtime setup and clock-lock state handling.
- `docker/rocm-targets.json` defines declared container target IDs, Docker tags,
  ROCm user-space versions, and wheel policies.
- Docker dependency checks live under `tests/docker/dependencies/`.

## Dataset And External Data Integrations

- Hugging Face dataset acquisition is represented by
  `scripts/download_data.sh` and `scripts/download_solexecbench.py`.
- Dataset migration CLI commands are implemented in `src/sol_execbench/cli/main.py`
  under the `dataset` Click group.
- Dataset layout, migration, manifest, inventory, readiness, and closure logic
  live under `src/sol_execbench/core/dataset/`.
- FlashInfer safetensors lookup roots are handled by
  `src/sol_execbench/core/bench/io.py` via `FLASHINFER_TRACE_DIR`.
- Downloaded source assets are expected under `data/` and should not be
  committed.

## CI And Repository Services

- GitHub Actions workflow `.github/workflows/code-quality.yml` runs Python
  quality gates on push and pull request events.
- `astral-sh/setup-uv@v6`, `actions/checkout@v4`, and
  `actions/setup-python@v5` are the external GitHub Actions used by CI.
- Pre-commit hooks are configured in `.pre-commit-config.yaml` and run local
  commands rather than remote hook repositories.

## No Application-Service Integrations

- No database, web server, auth provider, webhook receiver, or hosted API route
  is present in the repository.
- The project is a local CLI/package plus scripts and Docker tooling.
- Network use is limited to dependency indexes and optional dataset/download
  workflows.
