# Codebase Structure

**Analysis Date:** 2026-06-01

## Directory Layout

```text
SOL-ExecBench-ROCm/
├── src/sol_execbench/          # Installable Python package
│   ├── cli/                    # Click command entry points
│   ├── core/                   # Contracts, runtime services, dataset, scoring, reports
│   │   ├── bench/              # Evaluation runtime helpers and evidence collectors
│   │   │   └── config/         # Benchmark execution config dataclasses
│   │   ├── data/               # Pydantic schemas for public benchmark JSON
│   │   ├── dataset/            # Dataset manifests, readiness, closure, runner helpers
│   │   └── scoring/            # AMD SOL bound, SOLAR derivation, and score reports
│   ├── data/                   # Packaged static data
│   │   └── amd_hardware_models/ # AMD hardware model JSON data
│   └── driver/                 # Staging packager and generated evaluator templates
│       └── templates/          # Files copied into staging dirs
├── tests/                      # Pytest suite
│   ├── sol_execbench/          # Package tests and samples
│   ├── examples/               # Example workflow tests
│   ├── docker/dependencies/    # Container/runtime dependency tests
│   └── samples/                # Additional test-only sample inputs
├── examples/                   # Runnable example problems by backend/category
├── scripts/                    # Dataset, report, Docker, schema, and download CLIs
├── docs/                       # User, developer, schema, ROCm, and evidence docs
├── docker/                     # Dockerfile, entrypoint, and ROCm target manifest
├── data/                       # Downloaded benchmark assets and local data
├── .planning/                  # GSD project state, milestones, phase plans, codebase maps
├── .github/workflows/          # CI workflows
├── pyproject.toml              # Package metadata and tool config
├── uv.lock                     # Locked dependency graph
├── README.md                   # Project overview
├── CONTRIBUTING.md             # Contribution policy
├── SECURITY.md                 # Security policy
├── AGENTS.md                   # Agent instructions
└── CLAUDE.md                   # Claude-oriented agent instructions
```

## Directory Purposes

**`src/sol_execbench/`:**
- Purpose: Main package imported by console scripts, tests, and helper scripts.
- Contains: `cli/`, `core/`, `driver/`, packaged data, package exports, SOL score helper.
- Key files: `src/sol_execbench/__init__.py`, `src/sol_execbench/sol_score.py`.

**`src/sol_execbench/cli/`:**
- Purpose: User-facing CLIs.
- Contains: Main evaluator and baseline comparison commands.
- Key files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/cli/baseline.py`, `src/sol_execbench/cli/__init__.py`.

**`src/sol_execbench/core/data/`:**
- Purpose: Public benchmark contract models.
- Contains: Definitions, workloads, solutions, traces, evaluator contract, dtype conversion, shape expression resolution, JSON helpers, base model aliases.
- Key files: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/data/contract.py`, `src/sol_execbench/core/data/dtypes.py`, `src/sol_execbench/core/data/shapes.py`, `src/sol_execbench/core/data/json_utils.py`, `src/sol_execbench/core/data/base_model.py`.

**`src/sol_execbench/core/bench/`:**
- Purpose: Runtime benchmark implementation used by generated evaluator scripts and CLI sidecar collection.
- Contains: Input generation, safetensors loading, output allocation, correctness checks, latency timing, staged imports, reward-hack checks, clock locks, static evidence, ROCm profiling, timing policy.
- Key files: `src/sol_execbench/core/bench/eval_runtime.py`, `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/core/bench/clock_lock.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/bench/timing_policy.py`, `src/sol_execbench/core/bench/utils.py`.

**`src/sol_execbench/core/bench/config/`:**
- Purpose: Benchmark execution configuration.
- Contains: `BenchmarkConfig` and clock preset helpers.
- Key files: `src/sol_execbench/core/bench/config/benchmark_config.py`, `src/sol_execbench/core/bench/config/device_config.py`, `src/sol_execbench/core/bench/config/__init__.py`.

**`src/sol_execbench/core/dataset/`:**
- Purpose: Dataset-scale inspection, readiness, execution, and closure support.
- Contains: Category validation, checksums, layout inspection, manifests, inventory, readiness, ready subsets, run state, runner helpers, execution closure, run closure, paper denominator, parity gap, evidence refs.
- Key files: `src/sol_execbench/core/dataset/categories.py`, `src/sol_execbench/core/dataset/checksums.py`, `src/sol_execbench/core/dataset/layout.py`, `src/sol_execbench/core/dataset/manifest.py`, `src/sol_execbench/core/dataset/inventory.py`, `src/sol_execbench/core/dataset/readiness.py`, `src/sol_execbench/core/dataset/ready_subset.py`, `src/sol_execbench/core/dataset/run_state.py`, `src/sol_execbench/core/dataset/runner.py`, `src/sol_execbench/core/dataset/execution_closure.py`, `src/sol_execbench/core/dataset/run_closure.py`, `src/sol_execbench/core/dataset/paper_denominator.py`, `src/sol_execbench/core/dataset/parity_gap.py`, `src/sol_execbench/core/dataset/evidence_refs.py`.

**`src/sol_execbench/core/scoring/`:**
- Purpose: AMD-native scoring and derived SOL evidence.
- Contains: AMD hardware models, bound graph classification and estimates, SOL artifact builders, SOLAR derivation, score reports, baseline artifacts, bound sanity checks.
- Key files: `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_sol.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/amd_bound_classification.py`, `src/sol_execbench/core/scoring/amd_hardware_models.py`, `src/sol_execbench/core/scoring/baseline_artifact.py`, `src/sol_execbench/core/scoring/amd_bound_sanity.py`.

**`src/sol_execbench/core/`:**
- Purpose: Shared services outside the major subpackages.
- Contains: Public exports, environment diagnostics, toolchain routing, ROCm compatibility, Docker/dependency matrices, runtime evidence, baseline comparison, reporting, consistency, evaluation stability, matrix diff, trust summary, claim upgrade, score guardrails.
- Key files: `src/sol_execbench/core/__init__.py`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/compatibility.py`, `src/sol_execbench/core/docker_matrix.py`, `src/sol_execbench/core/dependency_matrix.py`, `src/sol_execbench/core/runtime_evidence.py`, `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/reporting.py`, `src/sol_execbench/core/consistency.py`, `src/sol_execbench/core/evaluation_stability.py`, `src/sol_execbench/core/matrix_diff.py`, `src/sol_execbench/core/trust_summary.py`, `src/sol_execbench/core/claim_upgrade.py`, `src/sol_execbench/core/scoring_guardrails.py`, `src/sol_execbench/core/diagnostics.py`.

**`src/sol_execbench/driver/`:**
- Purpose: Staging and generated execution boundary.
- Contains: `ProblemPackager`, package exports, generated templates.
- Key files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/__init__.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/templates/build_ext.py`.

**`src/sol_execbench/data/`:**
- Purpose: Packaged static data loaded by core services.
- Contains: AMD hardware model JSON files and package markers.
- Key files: `src/sol_execbench/data/amd_hardware_models/gfx1200.json`, `src/sol_execbench/data/amd_hardware_models/__init__.py`.

**`scripts/`:**
- Purpose: Repo-local operational CLIs around package helpers.
- Contains: Dataset runner, dataset downloaders, Docker wrapper, schema export, matrix diff, evidence and report commands.
- Key files: `scripts/run_dataset.py`, `scripts/run_docker.sh`, `scripts/download_solexecbench.py`, `scripts/download_data.sh`, `scripts/inspect_dataset.py`, `scripts/export_matrix_schema.py`, `scripts/report_amd_bound_sanity.py`, `scripts/report_claim_upgrade.py`, `scripts/report_consistency.py`, `scripts/report_evaluation_stability.py`, `scripts/report_paper_denominator.py`, `scripts/report_parity_gaps.py`, `scripts/report_trust_summary.py`, `scripts/diff_matrix_reports.py`.

**`examples/`:**
- Purpose: Runnable example benchmark problems and backend/category fixtures.
- Contains: Problem directories with `definition.json`, `workload.jsonl`, `reference.py`, solution JSON, Python kernels, HIP sources, and C++ sources.
- Key files: `examples/triton/rmsnorm/definition.json`, `examples/triton/rmsnorm/kernel.py`, `examples/hip_cpp/rmsnorm/kernel.hip`, `examples/hip_cpp/rmsnorm/main.cpp`, `examples/hipblas/gemm/main.cpp`, `examples/miopen/softmax/main.cpp`, `examples/rocwmma/gemm/kernel.hip`, `examples/ck/gemm/kernel.hip`.

**`tests/`:**
- Purpose: Pytest coverage for contracts, CLI behavior, driver staging, ROCm evidence, dataset workflows, scoring, docs, examples, and Docker dependencies.
- Contains: Package tests, driver tests, core tests, samples, fixtures, example tests, Docker dependency tests.
- Key files: `tests/conftest.py`, `tests/sol_execbench/test_e2e.py`, `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/core/bench/test_timing.py`, `tests/sol_execbench/core/data/test_solution.py`, `tests/examples/test_examples.py`, `tests/docker/dependencies/test_rocm_runtime.py`.

**`docs/`:**
- Purpose: User, developer, schema, ROCm, evidence, and release documentation.
- Contains: Getting started, testing, configuration, architecture, schema docs, ROCm timing/toolchain docs, evidence guides, internal validation notes, generated examples.
- Key files: `docs/ARCHITECTURE.md`, `docs/TESTING.md`, `docs/GETTING-STARTED.md`, `docs/CONFIGURATION.md`, `docs/rocm.md`, `docs/rocm_timing.md`, `docs/rocm_toolchain_routing.md`, `docs/static_kernel_evidence.md`, `docs/v1_19_evidence_guide.md`, `docs/v1_20_evidence_quality_guide.md`.

**`docker/`:**
- Purpose: ROCm evaluation container support.
- Contains: Dockerfile, entrypoint, target manifest.
- Key files: `docker/Dockerfile`, `docker/entrypoint.sh`, `docker/rocm-targets.json`.

**`.planning/`:**
- Purpose: GSD project state and generated planning context.
- Contains: Project documents, milestones, phase plans, quick work artifacts, codebase maps.
- Key files: `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`.

## Key File Locations

**Entry Points:**
- `src/sol_execbench/cli/main.py`: Main `sol-execbench` command and metadata subcommands.
- `src/sol_execbench/cli/baseline.py`: `sol-execbench-baseline` command.
- `scripts/run_dataset.py`: Dataset-scale benchmark execution.
- `scripts/run_docker.sh`: Docker environment wrapper.
- `src/sol_execbench/driver/templates/eval_driver.py`: Generated staging evaluator entry script.
- `src/sol_execbench/driver/templates/build_ext.py`: Generated staging native build entry script.

**Configuration:**
- `pyproject.toml`: Package metadata, dependencies, console scripts, pytest markers, Ruff and ty config, uv indexes.
- `uv.lock`: Locked dependency resolution.
- `.python-version`: Python version marker.
- `.pre-commit-config.yaml`: Pre-commit configuration.
- `docker/rocm-targets.json`: Docker ROCm target matrix.
- `src/sol_execbench/core/bench/config/benchmark_config.py`: Runtime benchmark config defaults.

**Core Logic:**
- `src/sol_execbench/core/data/definition.py`: Definition schema and axis/shape resolution.
- `src/sol_execbench/core/data/solution.py`: Solution schema, language policy, source path security, entry-point validation.
- `src/sol_execbench/core/data/workload.py`: Workload and input/tolerance schema.
- `src/sol_execbench/core/data/trace.py`: Canonical trace and evaluation schema.
- `src/sol_execbench/driver/problem_packager.py`: Staging directory builder.
- `src/sol_execbench/core/bench/eval_runtime.py`: Runtime import and timing helpers used by generated evaluator.
- `src/sol_execbench/core/bench/timing.py`: ROCm-compatible timing implementation.
- `src/sol_execbench/core/bench/correctness.py`: Correctness and tolerance checks.
- `src/sol_execbench/core/bench/reward_hack.py`: Static and runtime reward-hack guards.
- `src/sol_execbench/core/dataset/runner.py`: Importable dataset execution helpers.
- `src/sol_execbench/core/scoring/amd_score.py`: AMD-native score report construction.
- `src/sol_execbench/sol_score.py`: Anchored SOL score formula.

**Testing:**
- `tests/conftest.py`: Test fixtures and marker behavior.
- `tests/sol_execbench/`: Main package tests.
- `tests/sol_execbench/core/bench/`: Benchmark helper tests.
- `tests/sol_execbench/core/data/`: Contract tests.
- `tests/sol_execbench/driver/`: Staging and generated-driver tests.
- `tests/examples/`: Example validation tests.
- `tests/docker/dependencies/`: ROCm container dependency checks.

**Documentation:**
- `README.md`: High-level project entry.
- `docs/GETTING-STARTED.md`: Setup and first-run guidance.
- `docs/DEVELOPMENT.md`: Developer workflow.
- `docs/TESTING.md`: Test instructions.
- `docs/ARCHITECTURE.md`: User-facing architecture documentation.
- `docs/definition.md`, `docs/workload.md`, `docs/solution.md`, `docs/trace.md`: Public schema docs.
- `docs/rocm.md`, `docs/rocm_timing.md`, `docs/rocm_toolchain_routing.md`, `docs/rocm_libraries.md`: ROCm-specific docs.

## Naming Conventions

**Files:**
- Python modules use `snake_case.py`: `src/sol_execbench/core/dataset/run_state.py`, `src/sol_execbench/core/scoring/amd_bound_graph.py`.
- Tests use `test_*.py`: `tests/sol_execbench/test_e2e.py`, `tests/sol_execbench/core/bench/test_timing.py`.
- Example problem files use benchmark schema names: `definition.json`, `workload.jsonl`, `reference.py`, `solution_*.json`, `kernel.py`, `kernel.hip`, `main.cpp`.
- Generated or sidecar report files use descriptive suffixes in scripts and docs: `.environment.json`, `.profile.json`, `.static-evidence.json`, `.timing.json`, `.amd-sol-v2.json`, `.solar-derivation.json`.

**Directories:**
- Source package code belongs under `src/sol_execbench/`.
- Core subdomains belong under `src/sol_execbench/core/<domain>/`.
- Tests mirror source domains under `tests/sol_execbench/`, `tests/sol_execbench/core/`, and `tests/sol_execbench/driver/`.
- Runnable examples group by backend under `examples/<backend>/<problem>/`.
- Dataset assets belong under `data/`, not `src/` or `tests/`.

## Where to Add New Code

**New Evaluator CLI Option Or Subcommand:**
- Primary code: `src/sol_execbench/cli/main.py`
- Tests: `tests/sol_execbench/test_*.py`
- Docs: `docs/CONFIGURATION.md`, `docs/GETTING-STARTED.md`, or focused docs under `docs/`

**New Public Benchmark Schema Field:**
- Primary code: `src/sol_execbench/core/data/`
- Contract metadata: `src/sol_execbench/core/data/contract.py`
- Tests: `tests/sol_execbench/core/data/` and contract guardrails in `tests/sol_execbench/test_contract.py`
- Docs: `docs/definition.md`, `docs/workload.md`, `docs/solution.md`, or `docs/trace.md`

**New Runtime Evaluation Behavior:**
- Primary code: `src/sol_execbench/driver/templates/eval_driver.py` for generated evaluator flow.
- Shared helpers: `src/sol_execbench/core/bench/`
- Tests: `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/core/bench/`

**New Native Build/Staging Behavior:**
- Primary code: `src/sol_execbench/driver/problem_packager.py` or `src/sol_execbench/driver/templates/build_ext.py`
- Tests: `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/driver/test_build_ext.py`
- Example coverage: `examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/rocwmma/`, `examples/ck/`

**New Benchmark Helper:**
- Primary code: `src/sol_execbench/core/bench/`
- Tests: `tests/sol_execbench/core/bench/`
- Imports from generated evaluator: add only stable helpers needed by `src/sol_execbench/driver/templates/eval_driver.py`

**New Dataset Report Or Readiness Artifact:**
- Primary code: `src/sol_execbench/core/dataset/`
- Script wrapper: `scripts/report_*.py` or `scripts/inspect_dataset.py`
- Tests: `tests/sol_execbench/test_*report*.py`, `tests/sol_execbench/test_dataset_*.py`
- Docs: `docs/` or `docs/internal/`

**New Dataset Runner Behavior:**
- Primary code: `src/sol_execbench/core/dataset/runner.py` for reusable helpers and `scripts/run_dataset.py` for orchestration.
- Tests: `tests/sol_execbench/test_dataset_runner.py`, `tests/sol_execbench/test_run_dataset_*.py`

**New AMD Scoring Or SOL Bound Feature:**
- Primary code: `src/sol_execbench/core/scoring/`
- Packaged data: `src/sol_execbench/data/amd_hardware_models/` when hardware model data is needed.
- Tests: `tests/sol_execbench/test_amd_*.py`, `tests/sol_execbench/test_solar_derivation*.py`
- Docs: `docs/CLAIMS.md`, `docs/analysis.md`, or focused evidence docs under `docs/`

**New ROCm Toolchain Or Environment Probe:**
- Primary code: `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/dependency_matrix.py`, or `src/sol_execbench/core/docker_matrix.py`
- Tests: `tests/sol_execbench/test_environment_snapshot.py`, `tests/sol_execbench/test_toolchain_routing.py`, `tests/sol_execbench/test_dependency_matrix_*.py`, `tests/sol_execbench/test_docker_matrix_*.py`
- Docs: `docs/rocm.md`, `docs/rocm_toolchain_routing.md`, `docs/rocm_libraries.md`

**New Example Problem:**
- Primary code/data: `examples/<backend>/<problem>/definition.json`, `examples/<backend>/<problem>/workload.jsonl`, `examples/<backend>/<problem>/reference.py`, and solution files.
- Tests: `tests/examples/test_examples.py` or `tests/examples/test_rocm_cli_paths.py`
- Backend-specific examples: use ROCm-first categories such as `examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/rocwmma/`, `examples/ck/`, `examples/triton/`, `examples/pytorch/`.

**New Test Sample:**
- Package samples: `tests/sol_execbench/samples/`
- General samples: `tests/samples/`
- Fixtures: `tests/sol_execbench/fixtures/`

**Utilities:**
- Shared package helpers: `src/sol_execbench/core/utils.py` only for genuinely cross-cutting utilities.
- Domain helpers: keep helpers in the relevant subpackage such as `src/sol_execbench/core/dataset/` or `src/sol_execbench/core/scoring/`.

## Special Directories

**`.planning/`:**
- Purpose: GSD planning state and generated mapper output.
- Generated: Yes
- Committed: Project-dependent; codebase maps live in `.planning/codebase/`.

**`.artifacts/`:**
- Purpose: Local validation and evidence artifacts.
- Generated: Yes
- Committed: No for ordinary local output.

**`data/`:**
- Purpose: Downloaded benchmark datasets and local runtime data.
- Generated: Yes
- Committed: Only placeholders such as `data/.gitkeep`; downloaded assets stay uncommitted.

**`dist/`:**
- Purpose: Built source and wheel distributions.
- Generated: Yes
- Committed: No for ordinary rebuild output.

**`.uv-cache/`, `.ruff_cache/`, `.pytest_cache/`, `.venv/`:**
- Purpose: Tool caches and local virtual environment.
- Generated: Yes
- Committed: No.

**`src/sol_execbench/driver/templates/`:**
- Purpose: Runtime templates copied into staging directories.
- Generated: No
- Committed: Yes; changes affect evaluator subprocess behavior.

**`src/sol_execbench/data/amd_hardware_models/`:**
- Purpose: Packaged AMD hardware model data used by scoring.
- Generated: No
- Committed: Yes.

**`examples/`:**
- Purpose: Runnable benchmark examples and compatibility samples.
- Generated: No
- Committed: Yes, except local outputs.

---

*Structure analysis: 2026-06-01*
