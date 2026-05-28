# Codebase Structure

**Analysis Date:** 2026-05-28

## Directory Layout

```text
[project-root]/
+-- src/sol_execbench/        # Python package source
|   +-- cli/                  # Click command entry points
|   +-- core/                 # Schemas, benchmarking, scoring, dataset, environment logic
|   |   +-- bench/            # Runtime benchmark helpers and evidence collection
|   |   +-- data/             # Pydantic public contract models
|   |   +-- dataset/          # Dataset layout, inventory, readiness, manifest utilities
|   |   +-- scoring/          # AMD-native score and SOL-bound derivation
|   +-- data/                 # Packaged static data such as AMD hardware models
|   +-- driver/               # Staging packager and generated execution templates
+-- tests/                    # Pytest coverage and fixtures
|   +-- sol_execbench/        # Package-level unit/integration tests
|   +-- examples/             # Example workflow tests
|   +-- docker/               # Docker/dependency/runtime tests
|   +-- samples/              # Test-only sample solutions
+-- examples/                 # Runnable benchmark examples by solution category
+-- scripts/                  # Dataset, batch, Docker, and reporting scripts
+-- docs/                     # User, researcher, architecture, ROCm, and schema docs
+-- docker/                   # ROCm Dockerfile, entrypoint, target manifest
+-- data/                     # Downloaded benchmark assets
+-- .planning/                # GSD project planning and codebase maps
+-- pyproject.toml            # Packaging, dependencies, test markers, Ruff/Ty config
+-- uv.lock                   # Locked dependency resolution
```

## Directory Purposes

**`src/sol_execbench/`:**
- Purpose: Installable Python package for SOL ExecBench ROCm.
- Contains: Package exports, CLI, core domain logic, driver templates, packaged hardware-model data.
- Key files: `src/sol_execbench/__init__.py`, `src/sol_execbench/sol_score.py`, `src/sol_execbench/core/__init__.py`.

**`src/sol_execbench/cli/`:**
- Purpose: User-facing command layer.
- Contains: Main evaluator command and baseline comparison command.
- Key files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/cli/baseline.py`, `src/sol_execbench/cli/__init__.py`.

**`src/sol_execbench/core/`:**
- Purpose: Shared domain logic used by CLI, generated drivers, scripts, and tests.
- Contains: Public exports, baseline comparison helpers, diagnostics, environment snapshots, toolchain routing, Docker/dependency/runtime matrices, reporting, scoring guardrails.
- Key files: `src/sol_execbench/core/__init__.py`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/reporting.py`.

**`src/sol_execbench/core/data/`:**
- Purpose: Canonical public data schemas.
- Contains: Pydantic models for definitions, solutions, workloads, traces, evaluator contract, dtype helpers, shape resolution, JSON utilities.
- Key files: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/data/contract.py`.

**`src/sol_execbench/core/bench/`:**
- Purpose: Benchmark execution primitives.
- Contains: Input/output tensor utilities, correctness checks, timing, clock locking, reward-hack detection, ROCm profiler support, static kernel evidence.
- Key files: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`.

**`src/sol_execbench/core/bench/config/`:**
- Purpose: Runtime benchmark configuration and device clock presets.
- Contains: Benchmark config dataclass/model and device config helpers.
- Key files: `src/sol_execbench/core/bench/config/benchmark_config.py`, `src/sol_execbench/core/bench/config/device_config.py`.

**`src/sol_execbench/core/dataset/`:**
- Purpose: Dataset layout and readiness tooling.
- Contains: Category definitions, checksum helpers, inventory models, layout inspectors, manifests, parity gap reports, readiness classifiers, ready subset helpers.
- Key files: `src/sol_execbench/core/dataset/layout.py`, `src/sol_execbench/core/dataset/inventory.py`, `src/sol_execbench/core/dataset/manifest.py`, `src/sol_execbench/core/dataset/readiness.py`.

**`src/sol_execbench/core/scoring/`:**
- Purpose: AMD hardware/SOL-bound derivation and guarded score reporting.
- Contains: Bound estimates, bound graph, hardware models, AMD score reports, SOL derivation, baseline artifacts.
- Key files: `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_sol.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/baseline_artifact.py`, `src/sol_execbench/core/scoring/solar_derivation.py`.

**`src/sol_execbench/data/`:**
- Purpose: Package data distributed with the library.
- Contains: AMD hardware model JSON assets.
- Key files: `src/sol_execbench/data/amd_hardware_models/gfx1200.json`.

**`src/sol_execbench/driver/`:**
- Purpose: Isolated staging and generated driver scripts for compilation/evaluation.
- Contains: `ProblemPackager` and Python templates copied into temporary staging directories.
- Key files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/driver/templates/eval_driver.py`.

**`tests/`:**
- Purpose: Pytest test suite.
- Contains: Package tests, driver tests, core subpackage tests, example tests, Docker/dependency checks, sample benchmark data.
- Key files: `tests/conftest.py`, `tests/sol_execbench/test_e2e.py`, `tests/sol_execbench/driver/test_problem_packager.py`, `tests/examples/test_examples.py`, `tests/docker/dependencies/test_pytorch_rocm.py`.

**`examples/`:**
- Purpose: Runnable example problems organized by solution implementation family.
- Contains: `definition.json`, `workload.jsonl`, solution JSON files, reference code, and kernel sources.
- Key files: `examples/hip_cpp/rmsnorm/`, `examples/hipblas/gemm/`, `examples/miopen/softmax/`, `examples/triton/rmsnorm/`, `examples/pytorch/linear_backward/`.

**`scripts/`:**
- Purpose: Operational helpers outside the package console scripts.
- Contains: Dataset download/inspection, parity reporting, batch execution, Docker launcher.
- Key files: `scripts/run_dataset.py`, `scripts/download_solexecbench.py`, `scripts/inspect_dataset.py`, `scripts/report_parity_gaps.py`, `scripts/run_docker.sh`.

**`docs/`:**
- Purpose: User and project documentation.
- Contains: Architecture, configuration, development, testing, researcher guide, schema docs, ROCm docs, claim boundaries, release closures.
- Key files: `docs/ARCHITECTURE.md`, `docs/GETTING-STARTED.md`, `docs/TESTING.md`, `docs/rocm.md`, `docs/solution.md`, `docs/trace.md`.

**`docker/`:**
- Purpose: ROCm container support.
- Contains: Dockerfile, entrypoint, and target matrix manifest.
- Key files: `docker/Dockerfile`, `docker/entrypoint.sh`, `docker/rocm-targets.json`.

**`data/`:**
- Purpose: Local benchmark datasets and downloaded assets.
- Contains: Placeholder `.gitkeep`; generated/downloaded datasets belong here.
- Key files: `data/.gitkeep`.

**`.planning/`:**
- Purpose: GSD project planning artifacts and generated codebase intelligence.
- Contains: Project state, milestones, phases, research notes, codebase maps.
- Key files: `.planning/PROJECT.md`, `.planning/STATE.md`, `.planning/ROADMAP.md`, `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`.

## Key File Locations

**Entry Points:**
- `src/sol_execbench/cli/main.py`: Primary `sol-execbench` CLI and subcommand dispatcher.
- `src/sol_execbench/cli/baseline.py`: `sol-execbench-baseline` CLI.
- `scripts/run_dataset.py`: Batch runner over dataset problem directories.
- `scripts/download_solexecbench.py`: Hugging Face dataset acquisition script.
- `scripts/inspect_dataset.py`: Dataset inventory/readiness script.
- `scripts/run_docker.sh`: ROCm Docker build/run wrapper.
- `src/sol_execbench/core/docker_matrix.py`: Standalone CLI-style module for Docker target selection/preflight helpers.
- `src/sol_execbench/core/dependency_matrix.py`: Standalone CLI-style module for dependency policy observations.
- `src/sol_execbench/core/runtime_evidence.py`: Standalone CLI-style module for runtime evidence collection.

**Configuration:**
- `pyproject.toml`: Build backend, package metadata, dependencies, console scripts, pytest markers, Ruff excludes, Ty config, uv indexes.
- `uv.lock`: Dependency lockfile.
- `.python-version`: Python version pin for local tooling.
- `.pre-commit-config.yaml`: Pre-commit hooks.
- `docker/rocm-targets.json`: Declared ROCm Docker target matrix.
- `docker/Dockerfile`: ROCm container image build.
- `docker/entrypoint.sh`: Runtime environment setup in container.

**Core Logic:**
- `src/sol_execbench/core/data/definition.py`: Definition schema and reference-code validation.
- `src/sol_execbench/core/data/solution.py`: Solution schema, ROCm language categories, source path validation, compile options.
- `src/sol_execbench/core/data/workload.py`: Workload schema and input descriptors.
- `src/sol_execbench/core/data/trace.py`: Evaluation/trace schema and status invariants.
- `src/sol_execbench/core/bench/io.py`: Tensor generation, safetensors loading, output allocation/normalization.
- `src/sol_execbench/core/bench/correctness.py`: Numerical error and tolerance checks.
- `src/sol_execbench/core/bench/timing.py`: PyTorch ROCm event timing and memory-shifting allocator use.
- `src/sol_execbench/core/bench/reward_hack.py`: Static and runtime defenses against benchmark cheating.
- `src/sol_execbench/driver/problem_packager.py`: Staging directory contract and compile/evaluate command generation.
- `src/sol_execbench/driver/templates/eval_driver.py`: Generated runtime evaluator.
- `src/sol_execbench/driver/templates/build_ext.py`: Generated native extension builder.

**Scoring And Evidence:**
- `src/sol_execbench/core/scoring/amd_score.py`: Derived AMD-native workload and suite score reports.
- `src/sol_execbench/core/scoring/amd_sol.py`: AMD SOL bound artifact model.
- `src/sol_execbench/core/scoring/amd_sol_v2.py`: AMD SOL v2 bound artifact model.
- `src/sol_execbench/core/scoring/baseline_artifact.py`: Release scoring baseline artifact.
- `src/sol_execbench/core/scoring/solar_derivation.py`: SOLAR derivation evidence.
- `src/sol_execbench/core/environment.py`: Optional ROCm runtime environment snapshots and doctor diagnostics.
- `src/sol_execbench/core/bench/rocm_profiler.py`: Optional `rocprofv3` profile evidence.
- `src/sol_execbench/core/bench/static_kernel_evidence.py`: Optional static kernel evidence sidecars.
- `src/sol_execbench/core/toolchain.py`: Toolchain routing diagnostics.

**Testing:**
- `tests/conftest.py`: ROCm marker registration and skip policy.
- `tests/sol_execbench/core/data/`: Data model unit tests.
- `tests/sol_execbench/core/bench/`: Benchmark primitive tests.
- `tests/sol_execbench/driver/`: Packager/template tests.
- `tests/sol_execbench/test_e2e.py`: End-to-end CLI and evaluation tests.
- `tests/examples/test_examples.py`: Example problem tests.
- `tests/docker/dependencies/`: Container dependency/runtime checks.
- `tests/sol_execbench/samples/`: Test fixture problems and solution JSON files.

**Documentation:**
- `README.md`: Project overview and basic usage.
- `CONTRIBUTING.md`: Contribution workflow and DCO commit format.
- `docs/DEVELOPMENT.md`: Development guidance.
- `docs/TESTING.md`: Testing guidance.
- `docs/ARCHITECTURE.md`: User-facing architecture documentation.
- `docs/definition.md`, `docs/solution.md`, `docs/workload.md`, `docs/trace.md`: Public schema docs.
- `docs/rocm.md`, `docs/rocm_timing.md`, `docs/rocm_libraries.md`, `docs/rocm_toolchain_routing.md`: ROCm-specific docs.

## Naming Conventions

**Files:**
- Python modules use `snake_case.py`: `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/dataset/ready_subset.py`.
- Test files use `test_*.py`: `tests/sol_execbench/test_contract.py`, `tests/sol_execbench/core/bench/test_timing.py`.
- Example problem files use fixed benchmark names: `definition.json`, `workload.jsonl`, `solution_*.json`, `kernel.py`, `kernel.hip`, `main.cpp`, `reference.py`.
- Versioned JSON artifacts and sidecars use schema-specific names from the caller; keep deterministic JSON serialization in model/helper code.
- Generated driver templates are regular Python files under `src/sol_execbench/driver/templates/` and are copied by filename into staging directories.

**Directories:**
- Package subdirectories are domain nouns: `cli`, `core`, `driver`, `data`.
- Core subdirectories follow bounded responsibilities: `bench`, `dataset`, `scoring`, `data`.
- Examples are grouped by implementation category first, then operation: `examples/hip_cpp/rmsnorm`, `examples/hipblas/gemm`, `examples/triton/rmsnorm`.
- Tests mirror package boundaries where practical: `tests/sol_execbench/core/data`, `tests/sol_execbench/core/bench`, `tests/sol_execbench/driver`.
- Docker/dependency tests live under `tests/docker/dependencies`.

## Where to Add New Code

**New CLI Option Or Subcommand:**
- Primary code: `src/sol_execbench/cli/main.py` for evaluator metadata or execution behavior.
- Baseline-specific code: `src/sol_execbench/cli/baseline.py`.
- Tests: `tests/sol_execbench/test_contract.py`, `tests/sol_execbench/test_e2e.py`, or a focused `tests/sol_execbench/test_<feature>.py`.

**New Public Schema Field Or Model:**
- Primary code: `src/sol_execbench/core/data/`.
- Public export: `src/sol_execbench/core/data/__init__.py` and possibly `src/sol_execbench/core/__init__.py`.
- Docs: `docs/definition.md`, `docs/solution.md`, `docs/workload.md`, `docs/trace.md`, or `docs/CONFIGURATION.md`.
- Tests: `tests/sol_execbench/core/data/` and contract guardrails in `tests/sol_execbench/test_contract.py`.

**New Benchmark Runtime Behavior:**
- Input/output behavior: `src/sol_execbench/core/bench/io.py`.
- Correctness behavior: `src/sol_execbench/core/bench/correctness.py`.
- Timing behavior: `src/sol_execbench/core/bench/timing.py` and `src/sol_execbench/core/bench/timing_policy.py`.
- Reward-hack defense: `src/sol_execbench/core/bench/reward_hack.py`.
- Tests: `tests/sol_execbench/core/bench/` and end-to-end tests in `tests/sol_execbench/test_e2e.py`.

**New Generated Evaluation Logic:**
- Primary code: `src/sol_execbench/driver/templates/eval_driver.py`.
- Staging command/source changes: `src/sol_execbench/driver/problem_packager.py`.
- Native build changes: `src/sol_execbench/driver/templates/build_ext.py`.
- Tests: `tests/sol_execbench/driver/` plus focused E2E coverage.

**New Native ROCm Solution Category:**
- Schema category: `src/sol_execbench/core/data/solution.py`.
- Packager native-language set: `src/sol_execbench/driver/problem_packager.py`.
- Evaluation native-language set: `src/sol_execbench/driver/templates/eval_driver.py`.
- Examples: `examples/<category>/<operation>/`.
- Tests: `tests/examples/test_examples.py`, `tests/sol_execbench/test_rocm_library_examples.py`, and any dependency checks under `tests/docker/dependencies/`.

**New Evidence Sidecar:**
- Primary model/helper: a focused module under `src/sol_execbench/core/` or `src/sol_execbench/core/bench/`.
- CLI integration: sidecar path/write helper in `src/sol_execbench/cli/main.py`.
- Batch integration: `scripts/run_dataset.py` if dataset-scale collection is needed.
- Tests: focused tests under `tests/sol_execbench/` and docs guardrails if claims change.

**New AMD Scoring Or SOL-Bound Logic:**
- Primary code: `src/sol_execbench/core/scoring/`.
- Packaged static data: `src/sol_execbench/data/amd_hardware_models/` when distributing hardware model JSON.
- Tests: `tests/sol_execbench/test_amd_*.py`, `tests/sol_execbench/fixtures/solar_derivation/`.

**New Dataset Tooling:**
- Reusable logic: `src/sol_execbench/core/dataset/`.
- User script: `scripts/`.
- Tests: `tests/sol_execbench/test_dataset_*.py`, `tests/sol_execbench/test_download_solexecbench.py`, `tests/sol_execbench/test_parity_gap_report.py`.

**New Docker Or Dependency Matrix Behavior:**
- Target manifest: `docker/rocm-targets.json`.
- Docker support: `docker/Dockerfile`, `docker/entrypoint.sh`, `scripts/run_docker.sh`.
- Matrix logic: `src/sol_execbench/core/docker_matrix.py`, `src/sol_execbench/core/dependency_matrix.py`, `src/sol_execbench/core/runtime_evidence.py`.
- Tests: `tests/sol_execbench/test_docker_matrix_*.py`, `tests/sol_execbench/test_dependency_matrix_*.py`, `tests/docker/dependencies/`.

**Utilities:**
- Shared package helpers: `src/sol_execbench/core/utils.py` only for cross-cutting helpers that do not belong to a narrower module.
- Dataset-specific helpers: `src/sol_execbench/core/dataset/`.
- Scoring-specific helpers: `src/sol_execbench/core/scoring/`.
- CLI-only helpers: keep private in `src/sol_execbench/cli/main.py` or `src/sol_execbench/cli/baseline.py`.

## Special Directories

**`data/`:**
- Purpose: Downloaded benchmark datasets and generated dataset assets.
- Generated: Yes.
- Committed: Only placeholders or intentionally curated small files; downloaded datasets are not committed.

**`dist/`:**
- Purpose: Built wheel/source distributions.
- Generated: Yes.
- Committed: Existing build artifacts are present; do not add new build artifacts unless release workflow requires it.

**`.planning/`:**
- Purpose: GSD planning, milestone, phase, research, and codebase map artifacts.
- Generated: Yes.
- Committed: Yes, for planning artifacts managed by GSD.

**`.uv-cache/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`:**
- Purpose: Local tool caches.
- Generated: Yes.
- Committed: No.

**`examples/`:**
- Purpose: Source examples and compatibility fixtures.
- Generated: No.
- Committed: Yes.

**`tests/sol_execbench/samples/`:**
- Purpose: Test-only sample benchmark problems and solution fixtures.
- Generated: No.
- Committed: Yes.

**`src/sol_execbench/driver/templates/`:**
- Purpose: Runtime scripts copied into staging directories.
- Generated: No.
- Committed: Yes.

**Temporary `sol_execbench_*` staging directories:**
- Purpose: Per-run compile/evaluation workspaces created by `tempfile.mkdtemp()`.
- Generated: Yes.
- Committed: No.

---

*Structure analysis: 2026-05-28*
