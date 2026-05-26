# External Integrations

**Analysis Date:** 2026-05-26

## APIs & External Services

**Dataset Acquisition:**
- Hugging Face Datasets - Downloads the public `nvidia/SOL-ExecBench` dataset and writes local benchmark files.
  - SDK/Client: `datasets.load_dataset` from the `datasets` package in `scripts/download_solexecbench.py`.
  - Auth: None required by repository code; optional Hugging Face environment/CLI auth may be supplied externally for the `datasets` library.
- Hugging Face Hub CLI - Downloads `flashinfer-ai/flashinfer-trace` into `data/flashinfer-trace`.
  - SDK/Client: `hf download` invoked by `scripts/download_data.sh`; `docs/GETTING-STARTED.md` recommends `uv run --with "huggingface-hub[cli]" ./scripts/download_data.sh`.
  - Auth: None required by repository code; any Hugging Face token handling is external to this repo.

**Package Registries:**
- PyPI - Default Python package index.
  - SDK/Client: `uv` using the `pypi` index in `pyproject.toml`.
  - Auth: Not detected.
- PyTorch ROCm wheel indexes - Resolve ROCm-specific `torch`, `torchvision`, and `triton-rocm` wheels.
  - SDK/Client: `uv` using `https://download.pytorch.org/whl/rocm7.1` and `https://download.pytorch.org/whl/` configured in `pyproject.toml`.
  - Auth: Not detected.
- GitHub Container Registry - Provides the `uv` binary image during Docker builds.
  - SDK/Client: Docker `COPY --from=ghcr.io/astral-sh/uv:0.5.11` in `docker/Dockerfile`.
  - Auth: Not detected.
- Docker image registry for ROCm base image - Provides `rocm/dev-ubuntu-24.04:7.1.1-complete`.
  - SDK/Client: Docker build base image in `docker/Dockerfile`.
  - Auth: Not detected.

**Local ROCm Tooling:**
- ROCm runtime discovery tools - Probe device and runtime availability.
  - SDK/Client: `rocminfo`, `rocm_agent_enumerator`, `amd-smi`, and `rocm-smi` invoked from `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/diagnostics.py`, and `docker/entrypoint.sh`.
  - Auth: Local OS permissions for ROCm devices; no service auth.
- ROCm profiling tools - Collect optional diagnostic profiling artifacts.
  - SDK/Client: `rocprofv3` commands built and executed by `src/sol_execbench/core/bench/rocm_profiler.py` and exposed through `--profile rocprofv3` in `src/sol_execbench/cli/main.py`.
  - Auth: Local executable availability; no service auth.
- ROCm static evidence tools - Inspect native HIP/C++ build artifacts.
  - SDK/Client: `llvm-objdump`, `readelf`, optional `roc-objdump`, and planned `rga` modeled in `src/sol_execbench/core/toolchain.py` and used by static evidence code in `src/sol_execbench/core/bench/static_kernel_evidence.py`.
  - Auth: Local executable availability; no service auth.

## Data Storage

**Databases:**
- Not detected.
  - Connection: Not applicable.
  - Client: Not applicable.

**File Storage:**
- Local filesystem only.
  - Downloaded public benchmark data is written under `data/SOL-ExecBench/benchmark` by `scripts/download_solexecbench.py`.
  - FlashInfer trace assets are written under `data/flashinfer-trace` by `scripts/download_data.sh`.
  - CLI trace output and sidecars are written by `src/sol_execbench/cli/main.py` to user-provided `--output` paths, plus `.environment.json`, `.profile.json`, `.rocprofv3/`, and `.static-evidence` sidecars when enabled.
  - Temporary staging directories are created with `tempfile.mkdtemp(prefix="sol_execbench_")` in `src/sol_execbench/cli/main.py`.

**Caching:**
- `uv` cache is configured as `UV_CACHE_DIR=/home/${HOST_USER}/.cache/uv` in `docker/Dockerfile`.
- PyTorch allocation behavior is configured with `PYTORCH_ALLOC_CONF=expandable_segments:True` for compile/evaluation subprocesses in `src/sol_execbench/cli/main.py`.
- PyTorch and Triton runtime caches may be used by their libraries, but no external cache service is configured in this repository.

## Authentication & Identity

**Auth Provider:**
- Not detected.
  - Implementation: The project is a local CLI benchmark runner. There are no web sessions, users, tokens, OAuth flows, or application-level identity providers in `src/sol_execbench/`.

## Monitoring & Observability

**Error Tracking:**
- None.

**Logs:**
- CLI human output uses Rich tables/progress in `src/sol_execbench/cli/main.py`.
- Evaluation subprocess stdout is reserved for trace JSONL and stderr receives non-JSON output in `src/sol_execbench/driver/templates/eval_driver.py`.
- Runtime logs and failure messages are embedded in trace evaluation records through `src/sol_execbench/core/bench/utils.py`, `src/sol_execbench/core/reporting.py`, and `src/sol_execbench/core/data/trace.py`.
- Optional environment diagnostics are emitted by `sol-execbench doctor --json` from `src/sol_execbench/cli/main.py` using `src/sol_execbench/core/environment.py`.
- Optional profiler and static evidence sidecars are written by `src/sol_execbench/cli/main.py`.

## CI/CD & Deployment

**Hosting:**
- Not applicable. This repository packages a CLI and Dockerized local runner, not a hosted application.

**CI Pipeline:**
- GitHub Actions workflow `.github/workflows/code-quality.yml` runs on pull requests and pushes across Python `3.12` and `3.13`.
- The workflow installs with `uv sync --locked --all-groups`, then runs `uv run ruff check .`, `uv run ty check`, CPU-safe `pytest` under `tests/sol_execbench/`, and example consistency tests from `tests/examples/test_examples.py`.
- Local quality gates are `uv run pytest tests/`, Ruff, Ty, and Docker dependency smoke tests under `tests/docker/dependencies/`.
- `.pre-commit-config.yaml` defines local Ruff and DCO hooks.

## Environment Configuration

**Required env vars:**
- None for basic CLI usage.
- GPU/container execution expects ROCm device access and may use the following operational variables:
- `FLASHINFER_TRACE_DIR` - Optional safetensors lookup root used by `src/sol_execbench/driver/templates/eval_driver.py`.
- `SOL_EXECBENCH_CLOCKS_LOCKED` - Runtime state set by `docker/entrypoint.sh` and read by `src/sol_execbench/core/bench/clock_lock.py`.
- `SOL_EXECBENCH_SCLK_LEVEL` and `SOL_EXECBENCH_MCLK_LEVEL` - Optional clock-lock overrides read by `src/sol_execbench/core/bench/clock_lock.py`.
- `PYTORCH_ROCM_ARCH` - Optional native-build architecture override in `src/sol_execbench/driver/templates/build_ext.py`.
- `PYTORCH_ALLOC_CONF` - Set by `src/sol_execbench/cli/main.py` for subprocess allocator behavior.
- `SOLEXECBENCH_ENV_SNAPSHOT` and `SOLEXECBENCH_ENV_SNAPSHOT_PATH` - Optional environment snapshot controls in `src/sol_execbench/cli/main.py`.

**Secrets location:**
- Not detected.
- No `.env`, `.env.example`, `.env.sample`, `credentials.*`, or secret files were detected in the repository scan.
- `docs/CONFIGURATION.md` states there is no required application-level `.env` file.

## Webhooks & Callbacks

**Incoming:**
- None.

**Outgoing:**
- None.

---

*Integration audit: 2026-05-26*
