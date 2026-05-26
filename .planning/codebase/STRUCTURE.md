# Codebase Structure

**Analysis Date:** 2026-05-26

## Directory Layout

```text
SOL-ExecBench-ROCm/
├── src/sol_execbench/          # Python package source
│   ├── cli/                    # Click console commands
│   ├── core/                   # Contracts, benchmark runtime, dataset, scoring, diagnostics
│   │   ├── bench/              # Evaluation helpers used by generated drivers
│   │   │   └── config/         # Benchmark and device clock config
│   │   ├── data/               # Pydantic public schemas and JSON helpers
│   │   ├── dataset/            # Dataset layout, manifest, inventory, readiness, parity sidecars
│   │   └── scoring/            # AMD SOL/SOLAR/native score derivation
│   ├── data/                   # Packaged AMD hardware model data
│   │   └── amd_hardware_models/
│   └── driver/                 # Staging packager and generated subprocess templates
│       └── templates/
├── tests/                      # Pytest suite
│   ├── docker/dependencies/    # ROCm container dependency smoke tests
│   ├── examples/               # Example workflow tests
│   ├── samples/                # Test-only sample solution JSON
│   └── sol_execbench/          # Package tests
├── examples/                   # Runnable benchmark problem examples by solution category
├── scripts/                    # Dataset download, inspection, batch run, reporting helpers
├── docs/                       # User, researcher, ROCm, validation, and internal docs
├── docker/                     # ROCm Dockerfile and entrypoint
├── data/                       # Local downloaded benchmark assets
├── .github/workflows/          # GitHub Actions workflows
├── .planning/                  # GSD project planning and codebase maps
├── pyproject.toml              # Package, dependency, pytest, Ruff, ty, uv configuration
└── uv.lock                     # Locked dependency graph
```

## Directory Purposes

**`src/sol_execbench/`:**
- Purpose: Installable Python package exposed through console scripts and package imports.
- Contains: Package `__init__.py`, CLI, core contracts/runtime, packaged data, driver staging logic, `sol_score.py`.
- Key files: `src/sol_execbench/__init__.py`, `src/sol_execbench/sol_score.py`, `src/sol_execbench/core/__init__.py`.

**`src/sol_execbench/cli/`:**
- Purpose: User-facing Click commands.
- Contains: Main evaluator command and baseline comparison command.
- Key files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/cli/baseline.py`, `src/sol_execbench/cli/__init__.py`.

**`src/sol_execbench/core/data/`:**
- Purpose: Public JSON-compatible contracts for definitions, workloads, solutions, traces, evaluator metadata, dtypes, and shapes.
- Contains: Pydantic models and helpers.
- Key files: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/data/contract.py`.

**`src/sol_execbench/core/bench/`:**
- Purpose: Runtime support imported by generated evaluation drivers.
- Contains: Correctness, input/output generation, timing, clock locks, reward-hack detection, ROCm profiler/static evidence, timing policy, benchmark config.
- Key files: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/bench/config/benchmark_config.py`.

**`src/sol_execbench/core/dataset/`:**
- Purpose: Dataset asset inspection and sidecar generation.
- Contains: Category validation, checksums, layout inspection, manifests, inventory, readiness, ready subset, parity gap reports.
- Key files: `src/sol_execbench/core/dataset/layout.py`, `src/sol_execbench/core/dataset/manifest.py`, `src/sol_execbench/core/dataset/inventory.py`, `src/sol_execbench/core/dataset/readiness.py`, `src/sol_execbench/core/dataset/ready_subset.py`, `src/sol_execbench/core/dataset/parity_gap.py`.

**`src/sol_execbench/core/scoring/`:**
- Purpose: AMD-native score, SOL bounds, hardware model, bound graph, work estimate, and SOLAR derivation logic.
- Contains: Dataclasses/models and pure helpers that consume definitions, workloads, traces, baselines, and packaged hardware models.
- Key files: `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/amd_hardware_models.py`, `src/sol_execbench/core/scoring/amd_sol.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/solar_derivation.py`.

**`src/sol_execbench/core/`:**
- Purpose: Shared package-level services outside narrower subpackages.
- Contains: Baseline comparison, reporting, ROCm diagnostics/environment/toolchain helpers, score guardrails, package facade.
- Key files: `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/reporting.py`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/diagnostics.py`, `src/sol_execbench/core/scoring_guardrails.py`, `src/sol_execbench/core/utils.py`.

**`src/sol_execbench/driver/`:**
- Purpose: Runtime isolation boundary between trusted CLI and submitted solution code.
- Contains: `ProblemPackager` and generated build/evaluation templates copied into staging directories.
- Key files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/driver/templates/eval_driver.py`.

**`src/sol_execbench/data/`:**
- Purpose: Package data loaded by scoring/runtime helpers.
- Contains: AMD hardware model JSON and package markers.
- Key files: `src/sol_execbench/data/amd_hardware_models/gfx1200.json`, `src/sol_execbench/data/amd_hardware_models/__init__.py`.

**`tests/`:**
- Purpose: Pytest coverage for schemas, driver behavior, examples, ROCm toolchain checks, docs, dataset sidecars, scoring, and migration guardrails.
- Contains: Package tests, driver tests, docker dependency smoke tests, examples tests, sample solution files.
- Key files: `tests/conftest.py`, `tests/sol_execbench/test_e2e.py`, `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/driver/test_eval_driver.py`, `tests/examples/test_examples.py`, `tests/docker/dependencies/test_rocm_runtime.py`.

**`examples/`:**
- Purpose: Runnable problem examples organized by solution language/library category.
- Contains: Problem directories with `definition.json`, `workload.jsonl`, `reference.py`, `kernel.py` or native source, and `solution_*.json`.
- Key files: `examples/pytorch/gemma3_swiglu/solution_python.json`, `examples/triton/rmsnorm/solution_triton.json`, `examples/hip_cpp/rmsnorm/solution_hip.json`, `examples/hipblas/gemm/solution_hipblas.json`, `examples/miopen/softmax/solution_miopen.json`, `examples/ck/gemm/solution_ck.json`, `examples/rocwmma/gemm/solution_rocwmma.json`.

**`scripts/`:**
- Purpose: Operational tools around datasets and batch execution.
- Contains: Dataset download/import, dataset inspection, benchmark batch runner, parity gap reporting, Docker launcher.
- Key files: `scripts/run_dataset.py`, `scripts/inspect_dataset.py`, `scripts/download_solexecbench.py`, `scripts/report_parity_gaps.py`, `scripts/run_docker.sh`, `scripts/download_data.sh`.

**`docs/`:**
- Purpose: User-facing and internal project documentation.
- Contains: Architecture, configuration, getting started, testing, ROCm port notes, timing, tooling, solution/trace/schema docs, validation closures.
- Key files: `docs/ARCHITECTURE.md`, `docs/GETTING-STARTED.md`, `docs/TESTING.md`, `docs/CONFIGURATION.md`, `docs/rocm.md`, `docs/rocm_timing.md`, `docs/static_kernel_evidence.md`, `docs/rocm_toolchain_routing.md`.

**`docker/`:**
- Purpose: GPU evaluation container support.
- Contains: ROCm Dockerfile and entrypoint.
- Key files: `docker/Dockerfile`, `docker/entrypoint.sh`.

**`data/`:**
- Purpose: Local benchmark assets downloaded by helper scripts.
- Contains: Dataset roots and large local artifacts.
- Key files: Not committed source code; use `scripts/download_data.sh` and `scripts/download_solexecbench.py` to populate.

## Key File Locations

**Entry Points:**
- `src/sol_execbench/cli/main.py`: `sol-execbench` evaluator, metadata subcommands, environment/profile/static sidecar orchestration.
- `src/sol_execbench/cli/baseline.py`: `sol-execbench-baseline` command.
- `scripts/run_dataset.py`: Batch execution and score/closure sidecar orchestration.
- `scripts/inspect_dataset.py`: Dataset layout/inventory/readiness inspection.
- `scripts/download_solexecbench.py`: Dataset conversion/download helper.
- `scripts/report_parity_gaps.py`: Parity gap report generator.

**Configuration:**
- `pyproject.toml`: Build backend, package metadata, console scripts, runtime/dev dependencies, pytest markers/options, Ruff exclusions, ty include roots, uv indexes.
- `uv.lock`: Locked dependency resolution.
- `docker/Dockerfile`: ROCm container build.
- `docker/entrypoint.sh`: Container startup behavior.
- `.github/workflows/code-quality.yml`: CI quality workflow.
- `AGENTS.md`: Repository instructions and GSD context.

**Core Logic:**
- `src/sol_execbench/core/data/definition.py`: `Definition`, axes, tensor specs, dtype enum, reference validators.
- `src/sol_execbench/core/data/workload.py`: Workload rows, input descriptor union, tolerance.
- `src/sol_execbench/core/data/solution.py`: Solution schema, supported ROCm languages/hardware, compile options, migration validators.
- `src/sol_execbench/core/data/trace.py`: Trace, evaluation status, correctness/performance/environment records.
- `src/sol_execbench/core/bench/io.py`: Input generation, safetensors loading, output allocation/normalization.
- `src/sol_execbench/core/bench/correctness.py`: Numerical correctness metrics.
- `src/sol_execbench/core/bench/timing.py`: ROCm-compatible PyTorch device-event timing.
- `src/sol_execbench/core/bench/reward_hack.py`: Static and runtime anti-cheat checks.
- `src/sol_execbench/driver/problem_packager.py`: Staging, native compile command, eval command, trace parsing.
- `src/sol_execbench/driver/templates/eval_driver.py`: Generated evaluation subprocess.
- `src/sol_execbench/driver/templates/build_ext.py`: Generated native ROCm extension build subprocess.

**Dataset and Reporting:**
- `src/sol_execbench/core/dataset/layout.py`: Dataset root/category/problem layout inspection.
- `src/sol_execbench/core/dataset/inventory.py`: Dataset inventory records and denominators.
- `src/sol_execbench/core/dataset/readiness.py`: ROCm readiness classification.
- `src/sol_execbench/core/dataset/parity_gap.py`: Parity gap report model and markdown rendering.
- `src/sol_execbench/core/baseline.py`: Trace baseline load/compare/format helpers.
- `src/sol_execbench/core/reporting.py`: Trace summary and derived evidence report helpers.

**Scoring:**
- `src/sol_execbench/core/scoring/amd_bound_graph.py`: Bound graph construction and operator family classification.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py`: Operator work estimates.
- `src/sol_execbench/core/scoring/amd_hardware_models.py`: Hardware model loading and confidence enums.
- `src/sol_execbench/core/scoring/amd_sol.py`: AMD SOL bound calculations.
- `src/sol_execbench/core/scoring/amd_sol_v2.py`: SOL bound v2 artifact.
- `src/sol_execbench/core/scoring/solar_derivation.py`: SOLAR derivation sidecar evidence.
- `src/sol_execbench/core/scoring/amd_score.py`: AMD-native score report construction.
- `src/sol_execbench/sol_score.py`: Scalar SOL score helper.

**Testing:**
- `tests/sol_execbench/`: Primary package tests.
- `tests/sol_execbench/driver/`: Driver packager/template tests.
- `tests/examples/test_examples.py`: Example execution/schema coverage.
- `tests/docker/dependencies/`: ROCm and Python dependency smoke tests.
- `tests/samples/`: Sample malicious or edge-case solution JSON used by tests.

## Naming Conventions

**Files:**
- Python modules use `snake_case.py`: `src/sol_execbench/core/bench/timing_policy.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`.
- Tests use `test_*.py`: `tests/sol_execbench/test_toolchain_routing.py`, `tests/sol_execbench/driver/test_eval_driver.py`.
- Example problem files use fixed benchmark names: `definition.json`, `workload.jsonl`, `reference.py`, `kernel.py`, `solution_<category>.json`, optional native files such as `kernel.hip` or `main.cpp`.
- Documentation uses topic names in Markdown: `docs/rocm_timing.md`, `docs/static_kernel_evidence.md`, `docs/GETTING-STARTED.md`.
- Generated or operational JSON sidecars should use explicit subject names and remain outside package source unless they are package data.

**Directories:**
- Package subdirectories are domain nouns: `cli`, `core`, `bench`, `data`, `dataset`, `scoring`, `driver`.
- Example directories are grouped by solution category, then problem name: `examples/triton/rmsnorm/`, `examples/hip_cpp/rmsnorm/`.
- Tests mirror package or feature ownership where practical: `tests/sol_execbench/driver/` for `src/sol_execbench/driver/`, `tests/docker/dependencies/` for container dependency checks.
- Planning artifacts live under `.planning/`; codebase maps live under `.planning/codebase/`.

## Where to Add New Code

**New CLI Option or Subcommand:**
- Primary code: `src/sol_execbench/cli/main.py` for evaluator/metadata behavior, or `src/sol_execbench/cli/baseline.py` for baseline-specific behavior.
- Tests: `tests/sol_execbench/test_cli_environment_snapshot.py`, `tests/sol_execbench/test_toolchain_routing.py`, or a new focused `tests/sol_execbench/test_<feature>.py`.
- Keep user-facing parsing in `cli/`; put reusable behavior in `src/sol_execbench/core/`.

**New Benchmark Schema Field:**
- Primary code: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/solution.py`, or `src/sol_execbench/core/data/trace.py`.
- Tests: `tests/sol_execbench/test_contract.py`, `tests/sol_execbench/test_public_contract_guardrails.py`, and focused schema tests near related coverage.
- Update docs: `docs/definition.md`, `docs/workload.md`, `docs/solution.md`, or `docs/trace.md` as applicable.

**New Runtime Evaluation Behavior:**
- Primary code: `src/sol_execbench/core/bench/` for reusable helpers and `src/sol_execbench/driver/templates/eval_driver.py` only for generated subprocess orchestration.
- Tests: `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/test_e2e.py`, and a focused package test.
- Keep untrusted solution execution inside the generated driver subprocess.

**New Native ROCm Build Behavior:**
- Primary code: `src/sol_execbench/driver/problem_packager.py` for staging/command decisions and `src/sol_execbench/driver/templates/build_ext.py` for build execution.
- Tests: `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/driver/test_build_ext.py`, and `tests/docker/dependencies/` when ROCm toolchain behavior is involved.
- Example: add or update a matching example under `examples/<category>/<problem>/`.

**New Dataset Sidecar or Classification:**
- Primary code: `src/sol_execbench/core/dataset/`.
- Script integration: `scripts/inspect_dataset.py`, `scripts/download_solexecbench.py`, `scripts/report_parity_gaps.py`, or `scripts/run_dataset.py`.
- Tests: `tests/sol_execbench/test_dataset_inventory_readiness.py`, `tests/sol_execbench/test_dataset_contract.py`, or a new focused dataset test.

**New Scoring or Evidence Calculation:**
- Primary code: `src/sol_execbench/core/scoring/`.
- Integration: `scripts/run_dataset.py` only for CLI orchestration.
- Tests: `tests/sol_execbench/test_amd_bound_graph.py`, `tests/sol_execbench/test_amd_bound_estimates.py`, `tests/sol_execbench/test_amd_sol_v2.py`, `tests/sol_execbench/test_amd_native_score.py`, or focused SOLAR tests.
- Packaged hardware data: `src/sol_execbench/data/amd_hardware_models/`.

**New Environment, Diagnostics, or Toolchain Probe:**
- Primary code: `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/diagnostics.py`, or `src/sol_execbench/core/toolchain.py`.
- CLI integration: `src/sol_execbench/cli/main.py` metadata subcommands or sidecar writing.
- Tests: `tests/sol_execbench/test_environment_snapshot.py`, `tests/sol_execbench/test_rocm_diagnostics_reporting.py`, `tests/sol_execbench/test_toolchain_routing.py`.

**New Example Problem:**
- Implementation: `examples/<language_or_library>/<problem_name>/`.
- Required files: `definition.json`, `workload.jsonl`, `reference.py`, a solution JSON, and either `kernel.py` or native source files.
- Tests: `tests/examples/test_examples.py` and any category-specific test under `tests/sol_execbench/`.

**Utilities:**
- Shared helpers used by package code: `src/sol_execbench/core/utils.py` or a focused module under `src/sol_execbench/core/<domain>/`.
- One-off operational helpers: `scripts/`.
- Avoid importing from `scripts/` into package code.

## Special Directories

**`.planning/`:**
- Purpose: GSD project state, roadmap, phase artifacts, notes, and codebase maps.
- Generated: Yes.
- Committed: Project-dependent; current repo includes planning docs used by GSD workflows.

**`.artifacts/`:**
- Purpose: Local validation/evidence artifacts from benchmark and GSD runs.
- Generated: Yes.
- Committed: No for routine outputs unless a specific release artifact is intentionally tracked.

**`.uv-cache/`, `.ruff_cache/`, `.pytest_cache/`, `.venv/`, `dist/`:**
- Purpose: Tool caches, virtual environment, and build outputs.
- Generated: Yes.
- Committed: No.

**`data/`:**
- Purpose: Downloaded benchmark datasets and local benchmark assets.
- Generated: Yes.
- Committed: No for large/downloaded datasets.

**`src/sol_execbench/data/`:**
- Purpose: Small package data shipped with the Python package, currently AMD hardware model JSON.
- Generated: No.
- Committed: Yes.

**`src/sol_execbench/driver/templates/`:**
- Purpose: Python templates copied into staging directories and executed in subprocesses.
- Generated: No.
- Committed: Yes.
- Note: Treat these as runtime code; tests should cover template behavior because they execute outside the parent CLI process.

**`examples/`:**
- Purpose: Runnable sample problems and solution metadata across supported and migration-residue categories.
- Generated: No.
- Committed: Yes.

---

*Structure analysis: 2026-05-26*
