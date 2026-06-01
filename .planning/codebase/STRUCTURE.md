# Codebase Structure

**Analysis Date:** 2026-06-01

## Directory Layout

```text
SOL-ExecBench-ROCm/
├── src/sol_execbench/          # Python package source
│   ├── cli/                    # Click command entry points
│   ├── core/                   # Contracts, benchmark helpers, dataset, scoring, evidence, reports
│   │   ├── bench/              # Runtime benchmark helpers and optional evidence collection
│   │   ├── data/               # Public schemas for definitions, workloads, solutions, traces
│   │   ├── dataset/            # Dataset inventory/readiness/closure/parity helpers
│   │   └── scoring/            # AMD bound, SOL derivation, and AMD-native scoring helpers
│   ├── data/                   # Packaged static AMD hardware model data
│   └── driver/                 # Staging packager and generated runtime templates
├── tests/                      # Pytest suite
│   ├── sol_execbench/          # Package, schema, CLI, dataset, scoring, evidence tests
│   ├── examples/               # Example workflow tests
│   ├── docker/dependencies/    # ROCm container dependency tests
│   └── samples/                # Test-only problem and solution samples
├── examples/                   # Runnable benchmark examples by backend/category
├── scripts/                    # Dataset, Docker, report, and schema helper CLIs
├── docs/                       # User, developer, ROCm, schema, and release evidence docs
├── docker/                     # ROCm Dockerfile, entrypoint, and target manifest
├── data/                       # Downloaded benchmark assets and local runtime data
├── .planning/                  # GSD project state, milestones, phase plans, codebase maps
├── pyproject.toml              # Package metadata, dependencies, console scripts, pytest/Ruff config
├── uv.lock                     # Locked dependency resolution
├── README.md                   # Project overview
├── CONTRIBUTING.md             # Contribution policy
├── SECURITY.md                 # Security policy
└── AGENTS.md                   # Repository instructions for agents
```

## Directory Purposes

**`src/sol_execbench/`:**
- Purpose: Installable Python package for the ROCm SOL ExecBench evaluator.
- Contains: CLI entry points, core domain logic, static package data, driver staging code.
- Key files: `src/sol_execbench/__init__.py`, `src/sol_execbench/sol_score.py`.

**`src/sol_execbench/cli/`:**
- Purpose: User-facing command-line commands.
- Contains: Click commands for evaluation, metadata, toolchain diagnostics, and baseline comparison.
- Key files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/cli/baseline.py`, `src/sol_execbench/cli/__init__.py`.

**`src/sol_execbench/core/data/`:**
- Purpose: Public benchmark data contracts.
- Contains: Pydantic models and JSON helpers for definitions, workloads, solutions, traces, evaluator contract, shapes, dtypes, and base model behavior.
- Key files: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/data/contract.py`, `src/sol_execbench/core/data/json_utils.py`.

**`src/sol_execbench/core/bench/`:**
- Purpose: Runtime benchmark support used by the staged evaluation driver and CLI sidecar collection.
- Contains: Input/output allocation, correctness checks, timing, runtime imports, reward-hack guards, clock locks, profiler collection, static kernel evidence, benchmark config.
- Key files: `src/sol_execbench/core/bench/eval_runtime.py`, `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/bench/config/benchmark_config.py`.

**`src/sol_execbench/core/dataset/`:**
- Purpose: Dataset-level analysis and batch-run support.
- Contains: Category validation, checksums, manifests, layout diagnostics, inventory, readiness classification, ready subsets, run state, execution closure, paper denominator reports, parity gap reports, evidence-ref helpers.
- Key files: `src/sol_execbench/core/dataset/run_state.py`, `src/sol_execbench/core/dataset/inventory.py`, `src/sol_execbench/core/dataset/readiness.py`, `src/sol_execbench/core/dataset/execution_closure.py`, `src/sol_execbench/core/dataset/run_closure.py`, `src/sol_execbench/core/dataset/paper_denominator.py`, `src/sol_execbench/core/dataset/parity_gap.py`.

**`src/sol_execbench/core/scoring/`:**
- Purpose: AMD-native score and SOL-bound derivation logic.
- Contains: AMD hardware models, bound graph extraction, bound estimates, estimate families, SOL v1/v2 artifacts, solar derivation, AMD score reports, bound sanity checks, baseline scoring artifacts.
- Key files: `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_hardware_models.py`.

**`src/sol_execbench/core/`:**
- Purpose: Shared services that do not belong to the data, bench, dataset, or scoring subpackages.
- Contains: Environment diagnostics, toolchain routing, compatibility matrices, Docker/dependency matrices, runtime evidence, reporting, baseline comparison, consistency/stability/claim/trust reports, diagnostics, score guardrails, utility wrappers.
- Key files: `src/sol_execbench/core/__init__.py`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/compatibility.py`, `src/sol_execbench/core/docker_matrix.py`, `src/sol_execbench/core/dependency_matrix.py`, `src/sol_execbench/core/runtime_evidence.py`, `src/sol_execbench/core/reporting.py`, `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/trust_summary.py`.

**`src/sol_execbench/driver/`:**
- Purpose: Isolate user solution execution from the CLI process.
- Contains: `ProblemPackager` and templates copied into the staging directory.
- Key files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/templates/build_ext.py`.

**`src/sol_execbench/data/`:**
- Purpose: Package data shipped with `sol_execbench`.
- Contains: AMD hardware model JSON and package-data markers.
- Key files: `src/sol_execbench/data/amd_hardware_models/gfx1200.json`.

**`tests/`:**
- Purpose: Pytest coverage for contracts, CLI behavior, driver staging, dataset/reporting workflows, scoring, examples, and ROCm environment checks.
- Contains: Package tests in `tests/sol_execbench/`, example tests in `tests/examples/`, Docker dependency tests in `tests/docker/dependencies/`, shared fixtures/helpers.
- Key files: `tests/conftest.py`, `tests/sol_execbench_type_helpers.py`, `tests/sol_execbench/test_e2e.py`, `tests/examples/test_examples.py`, `tests/docker/dependencies/test_rocm_runtime.py`.

**`examples/`:**
- Purpose: Runnable example benchmark problems and solutions by backend.
- Contains: Problem directories with `definition.json`, `workload.jsonl`, reference code, kernel/source files, and solution JSON.
- Key directories: `examples/pytorch/`, `examples/triton/`, `examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/ck/`, `examples/rocwmma/`.

**`scripts/`:**
- Purpose: Operational helper scripts outside the package console-script surface.
- Contains: Dataset download/inspection/batch run tools, report generation, matrix schema export/diff, Docker launcher.
- Key files: `scripts/run_dataset.py`, `scripts/inspect_dataset.py`, `scripts/download_solexecbench.py`, `scripts/run_docker.sh`, `scripts/report_consistency.py`, `scripts/report_trust_summary.py`, `scripts/export_matrix_schema.py`.

**`docs/`:**
- Purpose: User docs, developer docs, schema docs, ROCm migration notes, static evidence/timing/toolchain guidance, and internal validation artifacts.
- Contains: Markdown docs and versioned evidence examples.
- Key files: `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md`, `docs/TESTING.md`, `docs/solution.md`, `docs/definition.md`, `docs/workload.md`, `docs/trace.md`, `docs/rocm.md`, `docs/rocm_timing.md`.

**`docker/`:**
- Purpose: ROCm-capable container support.
- Contains: Dockerfile, entrypoint, target manifest.
- Key files: `docker/Dockerfile`, `docker/entrypoint.sh`, `docker/rocm-targets.json`.

**`data/`:**
- Purpose: Local downloaded benchmark assets and runtime data.
- Contains: `.gitkeep`; downloaded datasets should stay local.
- Key files: `data/.gitkeep`.

**`.planning/`:**
- Purpose: GSD project state, requirements, roadmap, milestone plans, and codebase maps.
- Contains: Project metadata and generated planning artifacts.
- Key files: `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`.

## Key File Locations

**Entry Points:**
- `pyproject.toml`: Defines `sol-execbench = "sol_execbench.cli:cli"` and `sol-execbench-baseline = "sol_execbench.cli.baseline:cli"`.
- `src/sol_execbench/cli/main.py`: Main evaluator and metadata command implementation.
- `src/sol_execbench/cli/baseline.py`: Baseline comparison command implementation.
- `scripts/run_dataset.py`: Batch dataset runner entry point.
- `src/sol_execbench/core/dependency_matrix.py`: Also exposes a `main()` for dependency matrix CLI usage.
- `src/sol_execbench/core/docker_matrix.py`: Also exposes a `main()` for Docker matrix helper usage.
- `src/sol_execbench/core/runtime_evidence.py`: Also exposes a `main()` for runtime evidence helper usage.

**Configuration:**
- `pyproject.toml`: Build system, package metadata, runtime dependencies, dev dependencies, pytest markers, Ruff exclusions, uv indexes.
- `.python-version`: Local Python version hint.
- `.pre-commit-config.yaml`: Pre-commit hook configuration.
- `docker/rocm-targets.json`: ROCm Docker target manifest consumed by Docker matrix helpers.
- `src/sol_execbench/core/bench/config/benchmark_config.py`: Runtime benchmark configuration model.
- `src/sol_execbench/core/bench/config/device_config.py`: Device-related benchmark configuration.

**Core Logic:**
- `src/sol_execbench/core/data/`: Put schema/contract logic here.
- `src/sol_execbench/core/bench/`: Put code needed by `eval_driver.py` during live evaluation here.
- `src/sol_execbench/driver/`: Put staging and generated template logic here.
- `src/sol_execbench/core/dataset/`: Put dataset inventory, readiness, closure, and parity workflows here.
- `src/sol_execbench/core/scoring/`: Put AMD score, bound, graph, and derivation logic here.
- `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/compatibility.py`: Put ROCm evidence and compatibility models here.

**Testing:**
- `tests/sol_execbench/`: Put package and CLI tests here.
- `tests/sol_execbench/driver/`: Put focused driver/evaluator decomposition tests here.
- `tests/sol_execbench/core/`: Put focused core submodule tests here.
- `tests/examples/`: Put tests that verify runnable examples here.
- `tests/docker/dependencies/`: Put Docker/ROCm dependency checks here.
- `tests/samples/` and `tests/sol_execbench/samples/`: Put test-only sample problem files and malicious/edge-case solutions here.

**Documentation:**
- `docs/`: Put user and developer documentation here.
- `docs/internal/`: Put internal validation readiness and evidence notes here.
- `docs/examples/`: Put generated or curated report examples here.

## Naming Conventions

**Files:**
- Python modules use `snake_case.py`: `src/sol_execbench/core/runtime_evidence.py`, `src/sol_execbench/core/scoring/amd_bound_graph.py`.
- Tests use `test_*.py`: `tests/sol_execbench/test_e2e.py`, `tests/examples/test_examples.py`.
- Report scripts use verb-noun `snake_case.py`: `scripts/report_trust_summary.py`, `scripts/export_matrix_schema.py`.
- Benchmark problem files use fixed names: `definition.json`, `workload.jsonl`, `solution.json`, optional `config.json`.
- Generated sidecars use suffix-style names: `*.profile.json`, `*.static-evidence.json`, `*.environment.json`, `*.timing.json`.
- Native extension artifact name is fixed as `benchmark_kernel.so` in staging.

**Directories:**
- Package subpackages use short domain names: `cli`, `core`, `driver`, `data`, `bench`, `dataset`, `scoring`.
- Example backends use backend/category names: `examples/pytorch/`, `examples/triton/`, `examples/hip_cpp/`, `examples/ck/`.
- Dataset categories use source benchmark category names when downloaded under `data/`: `L1`, `L2`, `FlashInfer-Bench`, `Quant`.

## Where to Add New Code

**New CLI Option Or Metadata Subcommand:**
- Primary code: `src/sol_execbench/cli/main.py`
- Shared implementation: `src/sol_execbench/core/<domain>.py` or an existing subpackage
- Tests: `tests/sol_execbench/test_cli_*.py` or the closest existing CLI test file

**New Benchmark Runtime Behavior:**
- Primary code: `src/sol_execbench/core/bench/`
- Generated driver wiring: `src/sol_execbench/driver/templates/eval_driver.py`
- Tests: `tests/sol_execbench/driver/` for helper-level behavior and `tests/sol_execbench/test_e2e.py` for end-to-end behavior

**New Schema Field Or Public Contract:**
- Implementation: `src/sol_execbench/core/data/<schema>.py`
- Public exports: `src/sol_execbench/core/data/__init__.py` and, for common evaluator contracts, `src/sol_execbench/core/__init__.py`
- Documentation: matching file under `docs/`, such as `docs/solution.md`, `docs/definition.md`, `docs/workload.md`, or `docs/trace.md`
- Tests: `tests/sol_execbench/test_*contract*.py`, `tests/sol_execbench/test_public_contract_guardrails.py`, or a focused schema test

**New Solution Language Or Native Backend:**
- Schema support: `src/sol_execbench/core/data/solution.py`
- Build/staging support: `src/sol_execbench/driver/problem_packager.py` and possibly `src/sol_execbench/driver/templates/build_ext.py`
- Runtime loading support: `src/sol_execbench/core/bench/eval_runtime.py`
- Example: `examples/<backend>/<problem>/`
- Tests: `tests/examples/`, `tests/sol_execbench/test_rocm_library_examples.py`, and driver tests

**New Dataset Report Or Audit:**
- Primary code: `src/sol_execbench/core/dataset/` for dataset-state reports, or `src/sol_execbench/core/` for cross-report audits
- Script wrapper: `scripts/report_<name>.py`
- Tests: `tests/sol_execbench/test_<name>_report.py` and `tests/sol_execbench/test_<name>_script.py`
- Example output: `docs/examples/` when useful for docs or release evidence

**New Scoring Or Bound Logic:**
- Primary code: `src/sol_execbench/core/scoring/`
- Hardware data: `src/sol_execbench/data/amd_hardware_models/`
- Batch runner wiring: `scripts/run_dataset.py` if dataset runs should emit new sidecars
- Tests: `tests/sol_execbench/test_amd_*.py`, `tests/sol_execbench/test_solar_*.py`, or new focused scoring tests

**New Environment, Toolchain, Or Docker Evidence:**
- Environment probes: `src/sol_execbench/core/environment.py`
- Tool routing: `src/sol_execbench/core/toolchain.py`
- Compatibility matrix: `src/sol_execbench/core/compatibility.py`
- Docker target selection: `src/sol_execbench/core/docker_matrix.py`
- Dependency policy: `src/sol_execbench/core/dependency_matrix.py`
- Tests: `tests/sol_execbench/test_rocm_*.py`, `tests/sol_execbench/test_docker_*.py`, and `tests/docker/dependencies/`

**Utilities:**
- Schema JSON helpers: `src/sol_execbench/core/data/json_utils.py`
- Benchmark runtime helpers: `src/sol_execbench/core/bench/utils.py`
- General shared helpers: prefer a focused domain module in `src/sol_execbench/core/` over broad additions to `src/sol_execbench/core/utils.py`

## Special Directories

**`src/sol_execbench/driver/templates/`:**
- Purpose: Source templates copied into staging and executed there.
- Generated: No, but copied into generated staging directories.
- Committed: Yes.

**`data/`:**
- Purpose: Local downloaded benchmark assets and runtime data.
- Generated: Yes for downloaded contents.
- Committed: Only `data/.gitkeep`; do not commit datasets.

**`dist/`:**
- Purpose: Local package build artifacts.
- Generated: Yes.
- Committed: No.

**`.uv-cache/`, `.venv/`, `.ruff_cache/`, `.pytest_cache/`:**
- Purpose: Local tool caches and virtual environment state.
- Generated: Yes.
- Committed: No.

**`.artifacts/`:**
- Purpose: Local validation and evidence artifacts.
- Generated: Yes.
- Committed: No unless explicitly curated through docs or planning artifacts.

**`.planning/`:**
- Purpose: GSD planning state and generated codebase maps.
- Generated: Yes.
- Committed: Project-managed planning artifacts may be committed by the orchestrator.

**`examples/`:**
- Purpose: Curated runnable examples and ROCm migration examples.
- Generated: No.
- Committed: Yes.

---

*Structure analysis: 2026-06-01*
