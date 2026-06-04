---
generated_at: 2026-06-04
last_mapped_commit: ac6505f6511818160d36bc6935328ff0bd9468a6
focus: tech
scope: full repo
---

# Integrations

## GPU Runtime

- PyTorch ROCm is the primary runtime integration. The generated evaluation driver uses PyTorch `torch.cuda` APIs on ROCm, including the `"cuda:0"` device spelling, in `src/sol_execbench/driver/templates/eval_driver.py`.
- HIP/C++ builds integrate through `torch.utils.cpp_extension` in `src/sol_execbench/driver/templates/build_ext.py`.
- HIP-backed timing uses PyTorch device events and project timing policy helpers in `src/sol_execbench/core/bench/timing.py` and `src/sol_execbench/core/bench/timing_policy.py`.
- Runtime diagnostics collect PyTorch ROCm summary fields such as torch version, HIP version, device count, device name, and gfx target in `src/sol_execbench/core/environment.py`.

## ROCm System Tools

- `rocprofv3` is supported as optional diagnostic profiling evidence through `src/sol_execbench/core/bench/rocm_profiler.py` and the CLI `--profile rocprofv3` path.
- `rocprofv3-avail`, `rocminfo`, `rocm_agent_enumerator`, `amd-smi`, `rocm-smi`, `hipcc`, LLVM tools, `readelf`, and `llvm-objdump` are modeled as ROCm evidence/toolchain integrations.
- Toolchain registry and routing live in `src/sol_execbench/core/toolchain.py`.
- Static kernel evidence helpers live in `src/sol_execbench/core/bench/static_kernel_evidence.py`; evidence is diagnostic and not score authority.
- Optional clock-lock behavior integrates with `amd-smi` or `rocm-smi` through `src/sol_execbench/core/bench/clock_lock.py` and Docker entrypoint support.

## ROCm Libraries

- Accepted ROCm solution categories are declared in `src/sol_execbench/core/data/solution.py`:
  - `pytorch`
  - `triton`
  - `hip_cpp`
  - `hipblas`
  - `miopen`
  - `ck`
  - `rocwmma`
- Native/library examples are under `examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/ck/`, and `examples/rocwmma/`.
- Docker dependency checks cover HIP, ROCm runtime, PyTorch ROCm, Triton ROCm, and ROCm libraries under `tests/docker/dependencies/`.
- Legacy CUDA/NVIDIA category values such as `cuda_cpp`, `cublas`, `cudnn`, `cutlass`, `cute_dsl`, and `cutile` are rejected or retained only as migration/compatibility fixtures.

## External Data Sources

- Hugging Face Datasets integration is used by `scripts/download_solexecbench.py` through `datasets.load_dataset`.
- The public dataset identifier used by the downloader is `nvidia/SOL-ExecBench`.
- Dataset rows are transformed into local `definition.json`, `reference.py`, and `workload.jsonl` files under `data/SOL-ExecBench/benchmark/`.
- CLI dataset migration supports local-only migration for SOL ExecBench and FlashInfer Trace source layouts through `src/sol_execbench/core/dataset/migration.py`.
- FlashInfer Trace provenance is represented as `flashinfer-ai/flashinfer-trace` in migration and provenance metadata.
- Safetensors assets are referenced by workloads and loaded via the `safetensors` runtime dependency; downloaded benchmark blobs remain local data and are not committed.
- `Definition.hf_id` and dataset manifest source fields preserve optional Hugging Face/source metadata.

## Docker And Container Registry

- Docker builds start from ROCm development images such as `rocm/dev-ubuntu-24.04:<tag>`.
- The Dockerfile copies `uv` from `ghcr.io/astral-sh/uv:0.11.18`.
- Docker target metadata in `docker/rocm-targets.json` records ROCm user-space versions, PyTorch ROCm wheel policies, target IDs, and validation scope.
- `scripts/run_docker.sh` validates target identity, dependency matrix compatibility, GPU device access, and optional evidence sidecars before entering or running the container.
- Docker evidence is container user-space evidence; project docs avoid treating it as native-host validation.

## CI And Source Hosting

- GitHub Actions workflow `.github/workflows/code-quality.yml` integrates with `actions/checkout@v4`, `actions/setup-python@v5`, and `astral-sh/setup-uv@v6`.
- CI runs on `ubuntu-latest` for Python 3.12 and 3.13, then executes Ruff, Ty, and CPU-safe pytest subsets.
- Provenance metadata references upstream `https://github.com/NVIDIA/SOL-ExecBench` and FlashInfer Trace repository metadata.
- There are no application-level GitHub webhook handlers or GitHub API clients in the package.

## Release And Evidence Artifacts

- Prerelease artifact bundles are built by `scripts/build_prerelease_artifact_bundle.py`.
- Readiness gates are checked by `scripts/check_prerelease_readiness.py`.
- Release-candidate validation is implemented in `scripts/release_candidate_validation.py`.
- Dataset redistribution boundaries are enforced by `scripts/check_dataset_redistribution.py` and `provenance.toml`.
- Report scripts generate matrix schema, matrix diffs, parity gaps, AMD bound sanity, trust summary, consistency, claim-upgrade, evaluation-stability, and paper-denominator artifacts.

## Security And Isolation Boundaries

- Submitted solution code is staged and executed locally in a subprocess by the driver layer.
- The evaluator includes static source review and runtime reward-hack checks, but it is not a hardened sandbox or multi-tenant isolation boundary.
- Security guidance explicitly prohibits committing credentials, Hugging Face tokens, proprietary kernels, private datasets, or downloaded benchmark assets.
- Native compile options reject path injection, response files, runtime linker path control, and CUDA-specific compile option keys.

## Non-Integrations

- No database integration was found.
- No persistent backend service, hosted API server, web framework, or queue worker was found.
- No application authentication provider, OAuth flow, session store, webhook receiver, or payment provider was found.
- No cloud object storage integration such as S3 or GCS was found in application code.
- No external network calls are part of normal benchmark execution beyond explicit dependency installation, Docker image pulls, or explicit dataset download/migration workflows.
