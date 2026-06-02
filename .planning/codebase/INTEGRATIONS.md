---
generated_at: 2026-06-02
last_mapped_commit: 8019adc6295a78d4636037889245abcb3f9a52bb
focus: tech
---

# Integrations

## GPU Runtime

- PyTorch ROCm is the primary runtime integration. The generated evaluation driver uses `torch.cuda` APIs on ROCm, with `"cuda:0"` as the PyTorch device spelling in `src/sol_execbench/driver/templates/eval_driver.py`.
- HIP/C++ native builds route through PyTorch extension loading and project templates in `src/sol_execbench/driver/templates/build_ext.py`.
- HIP timing uses PyTorch/HIP-backed device events in `src/sol_execbench/core/bench/timing.py`.
- Optional clock policy is implemented through `src/sol_execbench/core/bench/clock_lock.py`.

## ROCm Tools

- `rocprofv3` profiling is optional diagnostic evidence, not correctness or score authority.
- Profiling helpers live in `src/sol_execbench/core/bench/rocm_profiler.py`.
- CLI profile routing is in `src/sol_execbench/cli/main.py` behind `--profile rocprofv3`.
- Toolchain registry and routing evidence live in `src/sol_execbench/core/toolchain.py`.
- Static kernel evidence uses routed extractor records in `src/sol_execbench/core/bench/static_kernel_evidence.py`.

## ROCm Libraries

- Supported native solution categories are declared in `src/sol_execbench/core/data/solution.py`: `hip_cpp`, `hipblas`, `miopen`, `ck`, and `rocwmma`.
- Public examples are under:
  - `examples/hip_cpp/`
  - `examples/hipblas/`
  - `examples/miopen/`
  - `examples/ck/`
  - `examples/rocwmma/`
- Docker dependency checks cover HIP, PyTorch ROCm, Triton ROCm, ROCm runtime, and ROCm libraries in `tests/docker/dependencies/`.

## External Data And Models

- Downloaded SOL ExecBench benchmark assets are expected under `data/`.
- `scripts/download_data.sh` and `scripts/download_solexecbench.py` handle dataset retrieval workflows.
- Dataset execution helpers in `scripts/run_dataset.py` discover benchmark directories with `definition.json` and `workload.jsonl`.
- Hugging Face-related metadata is represented by optional schema fields such as `Definition.hf_id` in `src/sol_execbench/core/data/definition.py`.

## Docker

- `docker/Dockerfile` builds from ROCm dev images and installs pinned PyTorch/Triton ROCm wheels.
- `scripts/run_docker.sh` validates target identity, dependency matrix compatibility, GPU device access, and optional compatibility sidecars.
- `docker/entrypoint.sh` sets runtime behavior inside the container.
- Docker evidence is documented as container ROCm user-space evidence, not native-host validation.

## Release Evidence

- Prerelease artifact bundles are built by `scripts/build_prerelease_artifact_bundle.py`.
- Readiness gates are checked by `scripts/check_prerelease_readiness.py`.
- Release candidate validation is in `scripts/release_candidate_validation.py`.
- Claim boundaries are documented in `docs/CLAIMS.md`, `docs/public_prerelease.md`, and `docs/research_preview.md`.
- Provenance and source attribution are governed by `provenance.toml` and `docs/provenance.md`.

## Non-Integrations

- There is no database, hosted API server, auth provider, webhook surface, or persistent backend service in the current codebase.
- NVIDIA/CUDA runtime paths are not maintained as a dual backend; legacy CUDA/NVIDIA schema values are rejected with ROCm migration guidance.
- CDNA4 validation is unavailable because suitable hardware is not currently accessible.
