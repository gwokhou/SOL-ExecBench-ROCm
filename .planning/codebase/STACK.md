---
generated_by: gsd-map-codebase
generated_on: 2026-07-09
last_mapped_commit: cc007cd3af3e5100f7d86f155a40d5e51ffb57e5
focus: tech
---

# Technology Stack

## Runtime

SOL ExecBench ROCm Port is a Python package under `src/sol_execbench/` requiring
Python `>=3.12,<3.14`. Packaging is defined in `pyproject.toml` with Hatchling
and `uv`-managed dependencies. The primary executable is `sol-execbench`,
implemented by `sol_execbench.cli.main:cli`; a second baseline comparison entry
point is `sol-execbench-baseline`.

## Core Dependencies

- `torch==2.10.0+rocm7.1` on Linux x86_64 via the `pytorch-rocm71` uv index.
- `torchvision==0.25.0+rocm7.1` on Linux x86_64 via the same ROCm index.
- `triton-rocm==3.6.0` from the PyTorch ROCm wheel root on Linux x86_64.
- `pydantic>=2.12.5` for all public schemas in `src/sol_execbench/core/data/`.
- `click>=8.0` and `rich>=13.0` for the CLI and terminal output.
- `datasets>=4.8.2`, `safetensors>=0.7.0`, `glom>=25.12.0`, `torch-c-dlpack-ext>=0.1.5`, and `apache-tvm-ffi>=0.1.9` for dataset, tensor, and integration workflows.
- `ninja>=1.13.0` for native extension builds through PyTorch.

Non-Linux or non-x86_64 platforms resolve to CPU/non-ROCm PyTorch wheels so
schema, docs, and CPU-safe tests can still run.

## GPU And Native Toolchain

The project targets ROCm `>=7.0`, with package metadata and Docker targets
currently aligned to ROCm 7.1 wheel sets. Native solution categories are built
through `torch.utils.cpp_extension` from staged HIP/C++ sources using
`src/sol_execbench/driver/templates/build_ext.py`.

Supported solution language categories are declared in
`src/sol_execbench/core/data/solution_models.py`:

- `pytorch`
- `triton`
- `hip_cpp`
- `hipblas`
- `miopen`
- `ck`
- `rocwmma`

The project intentionally rejects legacy CUDA/NVIDIA schema values such as
`cuda_cpp`, `cublas`, `cudnn`, `cutlass`, `cute_dsl`, and `cutile`.

## CLI And Application Framework

`Click` owns command parsing in `src/sol_execbench/cli/main.py` and
`src/sol_execbench/cli/commands/`. `Rich` is used for user-facing progress,
tables, and status output in the evaluator path. The root command dispatches
GPU-free metadata subcommands before falling back to evaluation.

Important command areas:

- `src/sol_execbench/cli/commands/root.py` for metadata dispatch.
- `src/sol_execbench/cli/commands/baseline.py` for trace baseline comparison and export.
- `src/sol_execbench/cli/commands/dataset.py` for local dataset migration commands.
- `src/sol_execbench/cli/commands/environment.py` and `src/sol_execbench/cli/commands/metadata.py` for diagnostics and contract output.
- `src/sol_execbench/cli/evaluation/` for the root evaluation workflow.

## Data And Validation

Pydantic v2 models define benchmark contracts:

- `src/sol_execbench/core/data/definition.py` and related `definition_*` modules.
- `src/sol_execbench/core/data/workload.py`.
- `src/sol_execbench/core/data/solution.py`, `solution_instance.py`, and `solution_models.py`.
- `src/sol_execbench/core/data/trace.py`.
- `src/sol_execbench/core/bench/config/benchmark_config.py`.

Models use project-specific base classes in `src/sol_execbench/core/data/base_model.py`
and include ROCm-specific validators for language categories, compile options,
source path traversal, target hardware, and entry-point compatibility.

## Scoring And Reports

Scoring logic lives in `src/sol_execbench/core/scoring/` and includes AMD-bound
estimates, AMD SOL artifacts, official score helpers, confidence helpers, and
solar derivation evidence parsing. Report builders live in
`src/sol_execbench/core/reports/` and cover consistency, matrix diffs, claim
upgrade evaluation, trust summaries, and evaluation stability.

## Dataset And Evidence Tooling

Dataset helpers live in `src/sol_execbench/core/dataset/` with subpackages for
inventory, migration, execution closure, paper denominator reporting, parity
gaps, readiness, sharding, and profiler timing coverage. Operator scripts in
`scripts/` and `scripts/internal/` call into these package modules rather than
duplicating all logic.

Evidence and sidecar logic is split across:

- `src/sol_execbench/core/evidence/` for runtime evidence references and collectors.
- `src/sol_execbench/core/bench/rocm_profiler/` for `rocprofv3` commands, artifacts, parsing, and timing.
- `src/sol_execbench/core/bench/static_kernel/` for static kernel evidence sidecars.
- `src/sol_execbench/core/bench/profile_summary/` for profile summary sidecars.
- `src/sol_execbench/core/bench/agent_feedback/` for bounded diagnostic feedback artifacts.

## Tooling

Development commands are defined by convention rather than a task runner:

- `uv sync --all-groups`
- `uv run pytest tests/`
- `uv run --with ruff ruff check .`
- `uv run --with ruff ruff format .`
- `uv run ty check`
- `./scripts/run_docker.sh --build`

CI is defined in `.github/workflows/code-quality.yml` and runs on Python 3.12
and 3.13 with `uv sync --locked --all-groups`, Ruff, Ty, and CPU-safe pytest
subsets.

## Container Stack

`docker/Dockerfile` builds the ROCm environment and verifies tool availability
including `hipcc`, `rocprofv3`, Python dependencies, PyTorch ROCm, Triton ROCm,
and ROCm library headers. `docker/rocm-targets.json` declares supported target
matrices and dependency versions. `scripts/run_docker.sh` wraps Docker build and
runtime invocation, including ROCm device passthrough and dependency argument
routing.
