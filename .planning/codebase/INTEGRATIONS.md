---
last_mapped_commit: f4de6692ee7468e150c112e0cbdcc8842dd0c709
mapped_at: 2026-06-06
---

# External Integrations

## Summary

This repository has no long-running web service, database, authentication
provider, webhook receiver, or hosted API server. External integrations are
primarily command-line, filesystem, package-index, dataset, Docker, and local GPU
toolchain integrations used to evaluate untrusted benchmark submissions on ROCm
hardware.

## Package And Build Indexes

`pyproject.toml` configures uv package indexes:

- PyPI through `https://pypi.org/simple`.
- PyTorch ROCm 7.1 wheels through `https://download.pytorch.org/whl/rocm7.1`.
- PyTorch wheel root through `https://download.pytorch.org/whl/`.

The Linux x86_64 runtime resolves ROCm wheels for `torch`,
`torchvision`, and `triton-rocm`. Non-Linux and non-x86_64 platforms resolve
non-ROCm PyTorch wheels for CPU-safe development tasks.

`docker/Dockerfile` also installs PyTorch ROCm and Triton ROCm wheels through
build args:

- `PYTORCH_ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm7.1`
- `TRITON_ROCM_INDEX_URL=https://download.pytorch.org/whl/`

The Dockerfile copies uv from `ghcr.io/astral-sh/uv:0.11.18`.

## Docker And Container Registry

The project integrates with ROCm Docker images:

- `docker/Dockerfile` defaults to base image `rocm/dev-ubuntu-24.04` with tag
  `7.1.1-complete`.
- `docker/rocm-targets.json` records container targets for `rocm/dev-ubuntu-24.04`
  tags `7.0.2-complete`, `7.1.1-complete`, and `7.2-complete`.
- `scripts/run_docker.sh` is the repo's Docker workflow entry point.

Container execution expects GPU device access to `/dev/kfd` and `/dev/dri`, as
documented in `README.md`. The Docker entrypoint in `docker/entrypoint.sh`
interacts with ROCm clock tools through sudo when available.

## Hugging Face Dataset Integration

`scripts/download_solexecbench.py` integrates with Hugging Face Datasets via
`datasets.load_dataset`.

The configured dataset repo ID is `nvidia/SOL-ExecBench`. The script downloads
selected public categories, converts rows into local `definition.json`,
`reference.py`, and `workload.jsonl` files, and writes them under
`data/SOL-ExecBench/benchmark/` by default.

The downloader supports an optional `--revision`, allowing callers to pin a
dataset revision. It can write deterministic acquisition/layout manifests using
`src/sol_execbench/core/dataset/manifest.py`.

`README.md` also references `scripts/download_data.sh` and
`huggingface-hub[cli]` for data acquisition workflows, but the active Python
downloader uses the `datasets` library directly.

## Local Dataset And File Integrations

The benchmark reads and writes local filesystem artifacts rather than calling a
remote service during evaluation.

Important local data surfaces:

- Problem inputs: `definition.json`, `workload.jsonl`, optional `config.json`, and
  solution JSON files.
- Downloaded benchmark data: `data/SOL-ExecBench/benchmark/`.
- Examples: `examples/`.
- Dataset run outputs and sidecars from `scripts/run_dataset.py`.
- Safetensors blobs referenced by workloads and loaded by
  `src/sol_execbench/core/bench/io.py`.

Safetensors lookup can use the `FLASHINFER_TRACE_DIR` environment variable. This
is handled by `src/sol_execbench/core/bench/io.py`, staged by
`src/sol_execbench/driver/problem_packager.py`, and checked in
`docker/entrypoint.sh`.

## ROCm Runtime And Hardware Tooling

The code integrates with local ROCm command-line tools and AMD GPU devices. These
are not web APIs; they are host/container executables.

Environment snapshot probes in `src/sol_execbench/core/environment.py` use:

- `amd-smi static -a`
- `rocminfo`
- `rocm_agent_enumerator`

HIP target detection in `src/sol_execbench/driver/problem_packager.py` uses:

- `rocm_agent_enumerator -name`
- `rocminfo`

Clock locking in `src/sol_execbench/core/bench/clock_lock.py` uses `rocm-smi`
through sudo when locking or resetting GPU clocks. `docker/entrypoint.sh` calls
this logic before running the requested command and resets clocks on exit.

ROCm device visibility and compatibility are captured through environment
variables in `src/sol_execbench/core/environment.py`, including
`HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `HSA_OVERRIDE_GFX_VERSION`.

## ROCm Profiling And Static Evidence Tools

Profiler integration is implemented in
`src/sol_execbench/core/bench/rocm_profiler.py` and exposed through the main CLI's
`--profile rocprofv3` option in `src/sol_execbench/cli/main.py`.

The profiler command built by the code invokes `rocprofv3` around the staged
application command and records diagnostic sidecars. These sidecars are explicitly
not score authority.

Toolchain routing metadata in `src/sol_execbench/core/toolchain.py` records
source references for ROCm tools, including AMD ROCm documentation and GitHub
repositories. Runtime routing can probe:

- `rocprofv3 --version`
- `rocprofv3-avail --help`

Static kernel evidence in
`src/sol_execbench/core/bench/static_kernel_evidence.py` collects local artifacts
from native HIP/C++ builds, runs bounded tool probes/extractors, writes sidecar
metadata, and marks the evidence diagnostic-only.

## Native Build Toolchain

Native HIP/C++ solutions are compiled locally through PyTorch C++ extension
support:

- `src/sol_execbench/driver/templates/build_ext.py` imports
  `torch.utils.cpp_extension`.
- `src/sol_execbench/driver/problem_packager.py` writes `build_ext.py` into the
  staging directory and returns `python build_ext.py` as the compile command.
- The expected output artifact is `benchmark_kernel.so`.

The Docker image installs system build tools, `ninja`, ROCm compiler tooling, and
ROCm libraries. The Dockerfile verifies the presence of `hipcc`, `rocminfo`,
`rocprofv3`, and either `amd-smi` or `rocm-smi` during image build.

Supported native solution categories include `hip_cpp`, `hipblas`, `miopen`,
`ck`, and `rocwmma`, defined in
`src/sol_execbench/core/data/solution.py` and exercised by example directories
under `examples/`.

## PyTorch ROCm And Triton ROCm

The evaluation runtime integrates with PyTorch ROCm and Triton ROCm locally:

- PyTorch ROCm availability and HIP version are probed in
  `src/sol_execbench/core/environment.py`.
- HIP-backed PyTorch events are used for timing in
  `src/sol_execbench/core/bench/timing.py`.
- Evaluation subprocesses set `PYTORCH_ALLOC_CONF=expandable_segments:True` in
  `src/sol_execbench/cli/main.py`.
- Triton ROCm is a supported solution language in
  `src/sol_execbench/core/data/solution.py` and appears in examples under
  `examples/triton/`.

PyTorch C++ extension dynamic loading is explicitly blocked inside the GPU
evaluation server by `src/sol_execbench/core/bench/eval_runtime.py` so submitted
Python code cannot build arbitrary dynamic extensions during evaluation.

## GitHub And CI Integrations

GitHub Actions workflow configuration is in `.github/workflows/code-quality.yml`.
It uses:

- `actions/checkout@v4`
- `actions/setup-python@v5`
- `astral-sh/setup-uv@v6`

The workflow runs package installation, Ruff, Ty, CPU-safe pytest coverage, and
example consistency checks on Python 3.12 and 3.13.

Pre-commit integration is configured in `.pre-commit-config.yaml` with local
hooks for Ruff, Ruff format, DCO sign-off enforcement, and Ty.

The contribution process references approved GitHub issues and DCO-signed commits
in `CONTRIBUTING.md` and `AGENTS.md`, but the application code itself does not
call the GitHub API.

## Documentation Source References

Several code and documentation paths include links to external source references
for claim boundaries and toolchain routing:

- `src/sol_execbench/core/toolchain.py` references ROCm docs and ROCm GitHub
  repositories for profiler/tool lifecycle metadata.
- `docs/rocm.md`, `docs/rocm_timing.md`, `docs/rocm_libraries.md`, and
  `docs/rocm_toolchain_routing.md` document ROCm operational assumptions.
- `provenance.toml` and `THIRD_PARTY_NOTICES.txt` track source and third-party
  obligations.

These are documentation/provenance integrations, not runtime service calls.

## Auth, Databases, And Webhooks

No database client, ORM, migration tool, auth provider, OAuth/OIDC flow, webhook
handler, message queue, or cloud SDK integration was found in the application
source.

Hugging Face dataset downloads may rely on the user's local Hugging Face
configuration if access is needed for a specific revision or cache state, but
`scripts/download_solexecbench.py` does not manage tokens or credentials.

No repository code should commit credentials, proprietary kernels, downloaded
datasets, or Hugging Face tokens. This boundary is documented in `AGENTS.md`,
`README.md`, and `SECURITY.md`.
