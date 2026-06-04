# Codebase Structure

**Analysis Date:** 2026-06-04

## Directory Layout

```text
SOL-ExecBench-ROCm/
├── src/sol_execbench/              # Installable Python package
│   ├── cli/                        # Click command entry points
│   ├── core/                       # Domain schemas, benchmark helpers, datasets, scoring, diagnostics
│   │   ├── bench/                  # Runtime evaluation helpers used by generated eval driver
│   │   ├── data/                   # Public Pydantic JSON/JSONL schemas
│   │   ├── dataset/                # Dataset migration, layout, run state, closure, runner helpers
│   │   └── scoring/                # AMD SOL bounds, score reports, baselines, derivation evidence
│   ├── data/                       # Packaged static data included with the package
│   └── driver/                     # Staging packager and generated execution/build templates
├── tests/                          # Pytest suite and sample problems
├── examples/                       # Runnable solution examples by backend/category
├── docs/                           # User, researcher, release, and internal documentation
├── scripts/                        # Dataset, Docker, report, release, and validation scripts
├── docker/                         # ROCm container definition, entrypoint, target manifest
├── data/                           # Downloaded benchmark assets; keep generated/downloaded data out of commits
├── dist/                           # Built distribution artifacts
├── .github/workflows/              # CI workflows
├── .planning/                      # GSD planning and codebase maps
├── pyproject.toml                  # Package metadata, console scripts, pytest and Ruff config
├── uv.lock                         # Locked dependency graph
├── README.md                       # Project overview and quick start
├── AGENTS.md                       # Repository agent instructions
└── LICENSE                         # Project license
```

## Directory Purposes

**`src/sol_execbench/`:**
- Purpose: Main installable package for the ROCm SOL ExecBench port.
- Contains: Public package exports, CLI modules, core services, driver templates, packaged AMD hardware model data.
- Key files: `src/sol_execbench/__init__.py`, `src/sol_execbench/sol_score.py`.

**`src/sol_execbench/cli/`:**
- Purpose: User-facing command-line interfaces.
- Contains: Root evaluator CLI and baseline comparison CLI.
- Key files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/cli/baseline.py`, `src/sol_execbench/cli/__init__.py`.

**`src/sol_execbench/core/`:**
- Purpose: Shared domain logic that must be importable by CLIs, scripts, tests, and generated driver code.
- Contains: Data contracts, benchmark runtime helpers, dataset helpers, scoring helpers, environment diagnostics, toolchain routing, reporting utilities.
- Key files: `src/sol_execbench/core/__init__.py`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/reporting.py`.

**`src/sol_execbench/core/data/`:**
- Purpose: Public JSON and JSONL schemas.
- Contains: `Definition`, `Workload`, `Solution`, `Trace`, dtype mappings, shape expression resolver, evaluator contract, JSON helpers.
- Key files: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/data/contract.py`.

**`src/sol_execbench/core/bench/`:**
- Purpose: Runtime mechanics for one benchmark evaluation.
- Contains: Input generation, output allocation, correctness checks, timing, reward-hack defenses, static kernel evidence, rocprofv3 profiling, clock locking, benchmark configuration.
- Key files: `src/sol_execbench/core/bench/eval_runtime.py`, `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/core/bench/config/benchmark_config.py`.

**`src/sol_execbench/core/dataset/`:**
- Purpose: Dataset-scale migration, inventory, execution, and evidence helpers.
- Contains: Categories, checksums, evidence references, execution closure, inventory, layout, migration, paper denominator and parity reports, readiness, run closure/state, runner, sharding.
- Key files: `src/sol_execbench/core/dataset/migration.py`, `src/sol_execbench/core/dataset/manifest.py`, `src/sol_execbench/core/dataset/runner.py`, `src/sol_execbench/core/dataset/run_state.py`, `src/sol_execbench/core/dataset/execution_closure.py`.

**`src/sol_execbench/core/scoring/`:**
- Purpose: AMD-native scoring and SOL-bound derivation.
- Contains: Bound classification, estimate families, hardware model loaders, graph extraction, AMD SOL v1/v2 artifacts, AMD-native suite reports, baseline artifacts, solar derivation evidence, sanity reports.
- Key files: `src/sol_execbench/core/scoring/amd_sol.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_hardware_models.py`, `src/sol_execbench/core/scoring/solar_derivation.py`.

**`src/sol_execbench/data/`:**
- Purpose: Package data loaded by scoring/runtime helpers.
- Contains: AMD hardware model JSON payloads.
- Key files: `src/sol_execbench/data/amd_hardware_models/gfx1200.json`.

**`src/sol_execbench/driver/`:**
- Purpose: Build the isolated staging directory used for subprocess evaluation.
- Contains: `ProblemPackager`, build/eval templates copied into staging.
- Key files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/templates/build_ext.py`.

**`tests/`:**
- Purpose: Pytest coverage for package behavior, scripts, docs contracts, examples, Docker helpers, and ROCm-sensitive paths.
- Contains: Top-level fixtures, package tests, example tests, sample problem directories, Docker dependency tests.
- Key files: `tests/conftest.py`, `tests/sol_execbench/`, `tests/examples/test_examples.py`, `tests/examples/test_rocm_cli_paths.py`.

**`examples/`:**
- Purpose: Reference solution examples grouped by implementation backend or migrated legacy category.
- Contains: `ck`, `cuda_cpp`, `cudnn`, `cute_dsl`, `cutile`, `cutlass`, `hip_cpp`, `hipblas`, `miopen`, `pytorch`, `rocwmma`, `triton`.
- Key files: `examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/triton/`, `examples/pytorch/`.

**`docs/`:**
- Purpose: Maintainer and user documentation for architecture, configuration, testing, ROCm support, scoring, provenance, releases, and evidence.
- Contains: Public guides, internal docs, release notes/checklists, example evidence directories.
- Key files: `docs/ARCHITECTURE.md`, `docs/CONFIGURATION.md`, `docs/TESTING.md`, `docs/rocm.md`, `docs/solution.md`, `docs/trace.md`.

**`scripts/`:**
- Purpose: Operational workflows outside the installable package API.
- Contains: Dataset runner, Docker runner, download helpers, report generators, release bundle/check scripts, validation scripts.
- Key files: `scripts/run_dataset.py`, `scripts/run_docker.sh`, `scripts/download_solexecbench.py`, `scripts/release_candidate_validation.py`, `scripts/build_prerelease_artifact_bundle.py`.

**`docker/`:**
- Purpose: ROCm-capable container build and runtime support.
- Contains: Dockerfile, entrypoint, ROCm target manifest.
- Key files: `docker/Dockerfile`, `docker/entrypoint.sh`, `docker/rocm-targets.json`.

**`.planning/`:**
- Purpose: GSD workflow state, project documents, milestone plans, quick work records, and generated codebase maps.
- Contains: Project roadmap/state, milestone artifacts, codebase docs, quick task directories.
- Key files: `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, `.planning/codebase/`.

## Key File Locations

**Entry Points:**
- `pyproject.toml`: Defines `sol-execbench = "sol_execbench.cli:cli"` and `sol-execbench-baseline = "sol_execbench.cli.baseline:cli"`.
- `src/sol_execbench/cli/__init__.py`: Re-exports the main CLI object.
- `src/sol_execbench/cli/main.py`: Root evaluator command and subcommands `contract`, `doctor`, `toolchain`, and `dataset`.
- `src/sol_execbench/cli/baseline.py`: Baseline comparison command.
- `scripts/run_dataset.py`: Dataset-scale runner.
- `scripts/run_docker.sh`: Docker environment runner.

**Configuration:**
- `pyproject.toml`: Package metadata, dependencies, console scripts, pytest markers, Ruff excludes.
- `uv.lock`: Dependency lockfile.
- `.python-version`: Python version pin for local tooling.
- `.pre-commit-config.yaml`: Pre-commit hook configuration.
- `docker/rocm-targets.json`: Docker ROCm target manifest.
- `src/sol_execbench/core/bench/config/benchmark_config.py`: Runtime benchmark defaults.
- `src/sol_execbench/core/bench/config/device_config.py`: Device clock presets.

**Core Logic:**
- `src/sol_execbench/core/data/definition.py`: Workload definition schema.
- `src/sol_execbench/core/data/workload.py`: Workload schema.
- `src/sol_execbench/core/data/solution.py`: Solution schema, language/hardware enums, source and compile boundary validation.
- `src/sol_execbench/core/data/trace.py`: Trace/evaluation schema.
- `src/sol_execbench/driver/problem_packager.py`: Staging and command construction.
- `src/sol_execbench/driver/templates/eval_driver.py`: Generated isolated evaluator.
- `src/sol_execbench/core/bench/eval_runtime.py`: Importable helpers used by eval driver.
- `src/sol_execbench/core/bench/timing.py`: PyTorch HIP event timing.
- `src/sol_execbench/core/bench/reward_hack.py`: Static and runtime benchmark-integrity checks.
- `src/sol_execbench/core/scoring/amd_score.py`: AMD-native score report construction.
- `src/sol_execbench/core/dataset/runner.py`: Dataset CLI invocation helpers.

**Testing:**
- `tests/conftest.py`: Shared pytest fixtures and ROCm/device marker behavior.
- `tests/sol_execbench/core/data/`: Schema tests.
- `tests/sol_execbench/core/bench/`: Benchmark helper tests.
- `tests/sol_execbench/driver/`: Staging/driver tests.
- `tests/sol_execbench/samples/`: Sample problem directories used by tests.
- `tests/examples/`: Example workflow tests.
- `tests/docker/`: Docker dependency and runtime preflight tests.

**Documentation:**
- `README.md`: Top-level project usage.
- `docs/ARCHITECTURE.md`: User-facing architecture documentation.
- `docs/GETTING-STARTED.md`: First-run instructions.
- `docs/CONFIGURATION.md`: Configuration and environment settings.
- `docs/TESTING.md`: Test strategy and commands.
- `docs/rocm.md`: ROCm support details.
- `docs/solution.md`, `docs/definition.md`, `docs/workload.md`, `docs/trace.md`: Public schema docs.

**Release and Evidence:**
- `docs/releases/`: Release notes.
- `docs/examples/`: Example evidence artifacts for docs/tests.
- `scripts/check_prerelease_readiness.py`: Prerelease readiness checks.
- `scripts/build_prerelease_artifact_bundle.py`: Artifact bundle assembly.
- `scripts/release_candidate_validation.py`: Release candidate validation.
- `.artifacts/`: Local generated evidence and run outputs; treat as generated/local unless explicitly curated.

## Naming Conventions

**Files:**
- Python modules use lowercase `snake_case.py`: `src/sol_execbench/core/scoring/amd_score.py`, `scripts/report_trust_summary.py`.
- Tests use `test_*.py`: `tests/sol_execbench/test_contract.py`, `tests/sol_execbench/core/bench/test_timing.py`.
- Markdown docs use uppercase for major guides or lowercase topic names according to existing docs: `docs/ARCHITECTURE.md`, `docs/rocm_timing.md`.
- JSON package data uses architecture or manifest names: `src/sol_execbench/data/amd_hardware_models/gfx1200.json`, `docker/rocm-targets.json`.

**Directories:**
- Package directories are lowercase and descriptive: `src/sol_execbench/core/dataset/`, `src/sol_execbench/core/scoring/`.
- Backend example directories use backend names: `examples/hip_cpp/`, `examples/hipblas/`, `examples/rocwmma/`, `examples/triton/`.
- Test sample directories identify the scenario/problem: `tests/sol_execbench/samples/rmsnorm/`, `tests/sol_execbench/samples/evil_monkey_patch/`.

**Classes and Models:**
- Use `PascalCase` for classes, dataclasses, and Pydantic models: `Definition`, `Solution`, `Trace`, `ProblemPackager`, `AmdNativeSuiteReport`.
- Use uppercase enum members with string values matching public schema values: `EvaluationStatus.PASSED`, `SupportedLanguages.HIP_CPP`.

**Functions and Variables:**
- Use `snake_case` for functions, methods, variables, and module constants with clear names.
- Use leading underscore for module-private helpers: `_load_definition`, `_inject_offload_arch_flags`, `_summarize_statistics`.
- Use all-caps for schema versions and policy constants: `AMD_SCORE_SCHEMA_VERSION`, `NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION`.

## Where to Add New Code

**New CLI Option For Single-Problem Evaluation:**
- Primary code: `src/sol_execbench/cli/main.py`
- Runtime behavior: `src/sol_execbench/core/bench/` or `src/sol_execbench/driver/templates/eval_driver.py` if it runs during evaluation.
- Tests: `tests/sol_execbench/test_e2e.py`, `tests/sol_execbench/core/bench/`, or focused CLI tests under `tests/sol_execbench/`.
- Docs: `docs/CONFIGURATION.md`, `docs/GETTING-STARTED.md`, or schema docs when user-visible.

**New `sol-execbench` Subcommand:**
- Primary code: Add a Click command in `src/sol_execbench/cli/main.py` and dispatch it through `SolExecbenchCli.main`.
- Shared logic: Add reusable functions/models under `src/sol_execbench/core/`.
- Tests: Add CLI tests under `tests/sol_execbench/` and core tests near the helper module.

**New Public Schema Field:**
- Primary code: `src/sol_execbench/core/data/`.
- Tests: `tests/sol_execbench/core/data/`.
- Docs: `docs/definition.md`, `docs/workload.md`, `docs/solution.md`, or `docs/trace.md`.
- Also update contract output in `src/sol_execbench/core/data/contract.py` when evaluator capability changes.

**New Benchmark Runtime Check:**
- Primary code: `src/sol_execbench/core/bench/`.
- Integration point: `src/sol_execbench/driver/templates/eval_driver.py`.
- Tests: `tests/sol_execbench/core/bench/` and a sample under `tests/sol_execbench/samples/` when behavior needs staged execution.

**New Native ROCm Solution Category:**
- Schema enum: `src/sol_execbench/core/data/solution.py`.
- Staging/compile category: `src/sol_execbench/driver/problem_packager.py`.
- Toolchain docs/routing: `src/sol_execbench/core/toolchain.py`, `docs/rocm_toolchain_routing.md`, `docs/rocm_libraries.md`.
- Examples: `examples/<category>/`.
- Tests: `tests/examples/`, `tests/sol_execbench/test_rocm_library_examples.py`, and focused schema/build tests.

**New Dataset Migration Or Inventory Behavior:**
- Reusable implementation: `src/sol_execbench/core/dataset/`.
- CLI exposure: `src/sol_execbench/cli/main.py` under the `dataset` group when user-facing.
- Batch script wiring: `scripts/run_dataset.py` when it is specific to dataset runs.
- Tests: `tests/sol_execbench/test_dataset_*` or a new focused test file under `tests/sol_execbench/`.

**New Scoring Or Evidence Artifact:**
- Primary code: `src/sol_execbench/core/scoring/`.
- Packaged constants/data: `src/sol_execbench/data/` when static data is part of the wheel.
- Scripts/reports: `scripts/report_*.py` or `scripts/release_candidate_validation.py`.
- Tests: `tests/sol_execbench/test_amd_*`, `tests/sol_execbench/test_solar_*`, or a focused scoring test file.

**New Report Script:**
- Primary code: `scripts/`.
- Shared computation: `src/sol_execbench/core/`.
- Tests: Add `tests/sol_execbench/test_<report_name>_script.py` and keep subprocess/file assertions focused.

**New Documentation:**
- Public guides: `docs/`.
- Internal notes: `docs/internal/`.
- Release-specific docs: `docs/releases/`.
- Planning artifacts: `.planning/` only through GSD workflows.

**Utilities:**
- Shared package helpers: `src/sol_execbench/core/utils.py` or a specific subpackage helper module.
- CLI-only helpers: Local private functions in `src/sol_execbench/cli/main.py` or `src/sol_execbench/cli/baseline.py`.
- Script-only helpers: Local private functions in the relevant file under `scripts/`.

## Special Directories

**`.planning/`:**
- Purpose: GSD project state and planning outputs.
- Generated: Yes.
- Committed: Project-dependent; edit only assigned planning files for mapper tasks.

**`.artifacts/`:**
- Purpose: Local run outputs, evidence, and generated validation artifacts.
- Generated: Yes.
- Committed: No by default unless a specific artifact is intentionally curated.

**`.uv-cache/`, `.ruff_cache/`, `.pytest_cache/`, `.venv/`:**
- Purpose: Local tool caches and virtual environment.
- Generated: Yes.
- Committed: No.

**`data/`:**
- Purpose: Downloaded or staged benchmark assets.
- Generated: Yes for most contents.
- Committed: Keep only lightweight placeholders or intentionally curated metadata.

**`dist/`:**
- Purpose: Built package artifacts.
- Generated: Yes.
- Committed: Usually not for source changes; existing artifacts may be present.

**`src/sol_execbench/driver/templates/`:**
- Purpose: Python files copied into evaluation staging directories.
- Generated: No in source tree; copied/generated at runtime in staging.
- Committed: Yes.
- Constraint: Code here runs as a script from a temporary directory; keep imports explicit and preserve stdout JSONL isolation.

**`src/sol_execbench/data/`:**
- Purpose: Package data shipped with the Python distribution.
- Generated: No for committed files.
- Committed: Yes.
- Constraint: Keep payloads small, deterministic, and covered by loader tests.

**`tests/sol_execbench/samples/`:**
- Purpose: Fixture problem directories and malicious-solution samples used by tests.
- Generated: No for committed fixtures.
- Committed: Yes.
- Constraint: Keep sample inputs minimal and deterministic.

---

*Structure analysis: 2026-06-04*
