# Codebase Structure

**Analysis Date:** 2026-05-22

## Directory Layout

```text
SOL-ExecBench-ROCm/
├── src/sol_execbench/          # Python package source
│   ├── cli/                    # Click command entry points
│   ├── core/                   # Schemas, benchmark utilities, scoring, reporting
│   │   ├── bench/              # Runtime benchmark helpers used by eval driver
│   │   │   └── config/         # Benchmark and clock preset config
│   │   ├── data/               # Public Pydantic data contracts
│   │   └── scoring/            # AMD SOL and suite scoring helpers
│   └── driver/                 # Staging packager and generated subprocess templates
│       └── templates/          # `eval_driver.py` and `build_ext.py`
├── tests/                      # Pytest suite
│   ├── sol_execbench/          # Package tests and sample benchmark assets
│   │   ├── core/               # Core data/bench tests
│   │   ├── driver/             # Packager and template tests
│   │   └── samples/            # Test problem fixtures
│   ├── examples/               # Example workflow tests
│   └── docker/dependencies/    # Container/runtime dependency checks
├── examples/                   # Runnable benchmark examples by language/library
├── docs/                       # User-facing and internal documentation
├── scripts/                    # Dataset, Docker, and download helpers
├── docker/                     # ROCm container files
├── data/                       # Local downloaded benchmark assets
├── .planning/                  # GSD planning artifacts and codebase maps
├── pyproject.toml              # Package, dependencies, pytest, Ruff, uv indexes
├── uv.lock                     # Locked dependency graph
├── README.md                   # Project overview
├── AGENTS.md                   # Agent/repository instructions
├── CONTRIBUTING.md             # Contribution process and commit format
└── SECURITY.md                 # Security policy
```

## Directory Purposes

**`src/sol_execbench/`:**
- Purpose: Installable Python package.
- Contains: Package exports, CLI entry points, core contracts/utilities, driver staging code.
- Key files: `src/sol_execbench/__init__.py`, `src/sol_execbench/sol_score.py`.

**`src/sol_execbench/cli/`:**
- Purpose: Console command layer.
- Contains: Click commands for benchmark execution and trace-baseline comparison.
- Key files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/cli/baseline.py`, `src/sol_execbench/cli/__init__.py`.

**`src/sol_execbench/core/data/`:**
- Purpose: Public benchmark schemas and JSON serialization helpers.
- Contains: Pydantic models for definitions, workloads, solutions, traces, shape/dtype helpers, JSON/JSONL utilities.
- Key files: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/data/json_utils.py`.

**`src/sol_execbench/core/bench/`:**
- Purpose: Runtime helpers used by the generated evaluation driver.
- Contains: Input/output tensor handling, correctness checks, timing, reward-hack defenses, ROCm profiler support, clock-lock checks.
- Key files: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/clock_lock.py`.

**`src/sol_execbench/core/bench/config/`:**
- Purpose: Execution configuration and hardware clock presets.
- Contains: `BenchmarkConfig`, `ClockPreset`, and device-name preset lookup.
- Key files: `src/sol_execbench/core/bench/config/benchmark_config.py`, `src/sol_execbench/core/bench/config/device_config.py`.

**`src/sol_execbench/core/scoring/`:**
- Purpose: AMD-native scoring, SOL bound artifacts, and baseline artifact support.
- Contains: Dataclasses and functions for AMD SOL bound estimation and guarded suite scores.
- Key files: `src/sol_execbench/core/scoring/amd_sol.py`, `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/baseline_artifact.py`.

**`src/sol_execbench/core/`:**
- Purpose: Shared core services outside data/bench/scoring subpackages.
- Contains: Public re-exports, baseline comparison, diagnostics, reporting, guardrails, environment utilities.
- Key files: `src/sol_execbench/core/__init__.py`, `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/diagnostics.py`, `src/sol_execbench/core/reporting.py`, `src/sol_execbench/core/scoring_guardrails.py`, `src/sol_execbench/core/utils.py`.

**`src/sol_execbench/driver/`:**
- Purpose: Filesystem staging and subprocess driver generation.
- Contains: `ProblemPackager`, package exports, and templates copied into staging.
- Key files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/templates/build_ext.py`.

**`tests/sol_execbench/`:**
- Purpose: Package tests and ROCm migration guardrails.
- Contains: End-to-end tests, public contract tests, scoring tests, docs guardrail tests, sample assets.
- Key files: `tests/sol_execbench/test_e2e.py`, `tests/sol_execbench/test_public_contract_guardrails.py`, `tests/sol_execbench/test_rocm_migration_residue_audit.py`.

**`tests/sol_execbench/core/`:**
- Purpose: Unit tests for core benchmark/data logic.
- Contains: `bench/` tests and `data/` tests.
- Key files: `tests/sol_execbench/core/bench/test_io.py`, `tests/sol_execbench/core/bench/test_timing.py`, `tests/sol_execbench/core/data/test_definition.py`, `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/core/data/test_workload.py`.

**`tests/sol_execbench/driver/`:**
- Purpose: Driver packager and generated template tests.
- Contains: Direct tests for staging, compilation template behavior, and `eval_driver.py`.
- Key files: `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/driver/test_build_ext.py`.

**`tests/sol_execbench/samples/`:**
- Purpose: Small benchmark fixtures for package tests.
- Contains: `definition.json`, `workload.jsonl`, solution JSON, and source files for test problems and reward-hack cases.
- Key files: `tests/sol_execbench/samples/rmsnorm/definition.json`, `tests/sol_execbench/samples/evil_monkey_patch/kernel.py`.

**`tests/examples/`:**
- Purpose: Tests that runnable examples package and execute correctly.
- Contains: Example discovery and packager execution checks.
- Key files: `tests/examples/test_examples.py`.

**`tests/docker/dependencies/`:**
- Purpose: Container/runtime dependency validation.
- Contains: Tests for HIP, PyTorch ROCm, Triton ROCm, ROCm runtime/libraries, and Python dependencies.
- Key files: `tests/docker/dependencies/test_hip.py`, `tests/docker/dependencies/test_pytorch_rocm.py`, `tests/docker/dependencies/test_triton_rocm.py`.

**`examples/`:**
- Purpose: Runnable problem examples grouped by solution language or library category.
- Contains: Example `definition.json`, `workload.jsonl`, `reference.py`, `solution_*.json`, and candidate source files.
- Key files: `examples/pytorch/linear_backward/`, `examples/triton/rmsnorm/`, `examples/hip_cpp/rmsnorm/`, `examples/hipblas/gemm/`.

**`docs/`:**
- Purpose: User-facing contracts, setup docs, and ROCm validation notes.
- Contains: Schema docs, architecture docs, solution/workload/trace docs, ROCm support docs, testing docs.
- Key files: `docs/ARCHITECTURE.md`, `docs/definition.md`, `docs/solution.md`, `docs/workload.md`, `docs/trace.md`, `docs/rocm.md`, `docs/TESTING.md`.

**`docs/internal/`:**
- Purpose: Internal inventories and validation readiness notes.
- Contains: ROCm compatibility and validation readiness documents.
- Key files: `docs/internal/v1_4_compatibility_inventory.md`, `docs/internal/cdna3_validation_readiness.md`, `docs/internal/mi300x_validation_readiness.md`.

**`scripts/`:**
- Purpose: Operational helpers.
- Contains: Dataset runner, dataset download scripts, and Docker launcher.
- Key files: `scripts/run_dataset.py`, `scripts/run_docker.sh`, `scripts/download_data.sh`, `scripts/download_solexecbench.py`.

**`docker/`:**
- Purpose: ROCm GPU evaluation container.
- Contains: Dockerfile and entrypoint.
- Key files: `docker/Dockerfile`, `docker/entrypoint.sh`.

**`data/`:**
- Purpose: Local benchmark assets downloaded by helper scripts.
- Contains: `.gitkeep`; generated/downloaded content should stay out of source edits.
- Key files: `data/.gitkeep`.

**`.planning/`:**
- Purpose: GSD project state, milestones, phase plans, and codebase maps.
- Contains: Project roadmap/state and generated analysis docs.
- Key files: `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, `.planning/codebase/`.

## Key File Locations

**Entry Points:**
- `src/sol_execbench/cli/main.py`: `sol-execbench` benchmark CLI.
- `src/sol_execbench/cli/baseline.py`: `sol-execbench-baseline` trace comparison CLI.
- `src/sol_execbench/cli/__init__.py`: CLI package export for `sol_execbench.cli:cli`.
- `src/sol_execbench/__init__.py`: Package-level public API exports.
- `src/sol_execbench/core/__init__.py`: Core model/config re-exports.
- `scripts/run_dataset.py`: Batch dataset runner.
- `scripts/run_docker.sh`: Docker build/run wrapper.
- `docker/entrypoint.sh`: Container startup script.

**Configuration:**
- `pyproject.toml`: Build backend, package metadata, dependencies, script entry points, pytest markers, Ruff excludes, and uv ROCm indexes.
- `uv.lock`: Locked dependencies.
- `.python-version`: Local Python version hint.
- `.pre-commit-config.yaml`: Pre-commit hook configuration.
- `docker/Dockerfile`: ROCm container image.
- `src/sol_execbench/core/bench/config/benchmark_config.py`: Runtime benchmark defaults.
- `src/sol_execbench/core/bench/config/device_config.py`: GPU clock preset mapping.

**Core Logic:**
- `src/sol_execbench/core/data/definition.py`: Problem definition schema and axis/shape resolution.
- `src/sol_execbench/core/data/workload.py`: Workload input and tolerance schema.
- `src/sol_execbench/core/data/solution.py`: Solution/build schema, supported ROCm languages, hardware targets, source validation.
- `src/sol_execbench/core/data/trace.py`: Trace/evaluation schema and status validation.
- `src/sol_execbench/core/bench/io.py`: Input generation, safetensors loading, output normalization/allocation, memory pool allocator.
- `src/sol_execbench/core/bench/correctness.py`: Seed control and numerical correctness metrics.
- `src/sol_execbench/core/bench/timing.py`: ROCm-compatible PyTorch device event timing.
- `src/sol_execbench/core/bench/reward_hack.py`: Static/runtime reward-hack detection.
- `src/sol_execbench/driver/problem_packager.py`: Staging directory writer and subprocess command builder.
- `src/sol_execbench/driver/templates/eval_driver.py`: Generated evaluation runtime.
- `src/sol_execbench/driver/templates/build_ext.py`: Generated HIP/C++ extension build runtime.

**Scoring and Reporting:**
- `src/sol_execbench/core/baseline.py`: Trace baseline comparison.
- `src/sol_execbench/core/reporting.py`: Trace run and evidence report summaries.
- `src/sol_execbench/core/scoring/amd_sol.py`: AMD SOL bound estimation.
- `src/sol_execbench/core/scoring/amd_score.py`: AMD-native score report construction.
- `src/sol_execbench/core/scoring/baseline_artifact.py`: Scoring baseline artifact loading.
- `src/sol_execbench/core/scoring_guardrails.py`: Score interpretation guardrails.
- `src/sol_execbench/sol_score.py`: Scalar SOL score formula.

**Testing:**
- `tests/conftest.py`: Shared pytest configuration.
- `tests/sol_execbench/core/data/`: Schema/model unit tests.
- `tests/sol_execbench/core/bench/`: Benchmark utility unit tests.
- `tests/sol_execbench/driver/`: Driver and template tests.
- `tests/sol_execbench/test_e2e.py`: End-to-end package/driver tests.
- `tests/examples/test_examples.py`: Runnable example coverage.
- `tests/docker/dependencies/`: ROCm container dependency checks.

**Documentation:**
- `docs/ARCHITECTURE.md`: User-facing architecture overview.
- `docs/GETTING-STARTED.md`: Setup and first-run instructions.
- `docs/CONFIGURATION.md`: Benchmark configuration reference.
- `docs/definition.md`, `docs/workload.md`, `docs/solution.md`, `docs/trace.md`: Public schema documentation.
- `docs/rocm.md`, `docs/rocm_timing.md`, `docs/rocm_libraries.md`: ROCm-specific runtime and library guidance.
- `docs/TESTING.md`: Test command and marker guidance.

## Naming Conventions

**Files:**
- Python modules use `snake_case.py`: `problem_packager.py`, `benchmark_config.py`, `scoring_guardrails.py`.
- Tests use `test_*.py`: `tests/sol_execbench/driver/test_problem_packager.py`.
- JSON benchmark assets use fixed contract names: `definition.json`, `workload.jsonl`, `config.json`, `solution.json` or `solution_<language>.json`.
- Generated/staged driver templates keep executable names: `eval_driver.py`, `build_ext.py`.
- Documentation uses topic names in Markdown: `docs/solution.md`, `docs/CONFIGURATION.md`, `docs/ARCHITECTURE.md`.

**Directories:**
- Python package directories use lowercase snake case or compact package names: `src/sol_execbench/core/bench/`.
- Example directories are grouped by language/library, then problem name: `examples/hip_cpp/rmsnorm/`, `examples/triton/olmo3_post_norm/`.
- Test sample directories mirror benchmark problem names: `tests/sol_execbench/samples/gqa_paged_decode/`.

## Where to Add New Code

**New CLI Option or Command:**
- Primary code: `src/sol_execbench/cli/main.py` for benchmark execution, or `src/sol_execbench/cli/baseline.py` for trace comparison.
- Tests: `tests/sol_execbench/test_e2e.py` for command behavior, plus focused tests under `tests/sol_execbench/` when the option affects contracts.
- Docs: `docs/GETTING-STARTED.md`, `docs/CONFIGURATION.md`, or related schema docs when user-facing behavior changes.

**New Public Schema Field:**
- Primary code: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/solution.py`, or `src/sol_execbench/core/data/trace.py`.
- Tests: Matching file under `tests/sol_execbench/core/data/`.
- Docs: `docs/definition.md`, `docs/workload.md`, `docs/solution.md`, or `docs/trace.md`.
- Public exports: Update `src/sol_execbench/core/data/__init__.py`, `src/sol_execbench/core/__init__.py`, and `src/sol_execbench/__init__.py` when adding exported model types.

**New Benchmark Runtime Behavior:**
- Primary code: `src/sol_execbench/core/bench/` for reusable helpers and `src/sol_execbench/driver/templates/eval_driver.py` for subprocess runtime wiring.
- Tests: `tests/sol_execbench/core/bench/` for helper behavior and `tests/sol_execbench/driver/test_eval_driver.py` for staged runtime behavior.

**New Native ROCm Language or Library Category:**
- Primary code: `SupportedLanguages` and validators in `src/sol_execbench/core/data/solution.py`.
- Driver support: Add language to `_CPP_LANGUAGES` in `src/sol_execbench/driver/problem_packager.py` and `_NATIVE_ROCM_LANGUAGES` in `src/sol_execbench/driver/templates/eval_driver.py` when it compiles/imports as native code.
- Examples: Add runnable example under `examples/<language>/<problem>/`.
- Tests: Add schema/packager/example tests under `tests/sol_execbench/` and `tests/examples/`.
- Docs: Update `docs/solution.md`, `docs/rocm_libraries.md`, and relevant ROCm docs.

**New Input Generation or Correctness Rule:**
- Primary code: `src/sol_execbench/core/bench/io.py` for generated inputs and output handling; `src/sol_execbench/core/bench/correctness.py` for numerical comparisons.
- Tests: `tests/sol_execbench/core/bench/test_io.py` or `tests/sol_execbench/core/bench/test_correctness.py`.

**New Timing or Profiling Behavior:**
- Primary code: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/timing_policy.py`, or `src/sol_execbench/core/bench/rocm_profiler.py`.
- Tests: `tests/sol_execbench/core/bench/test_timing.py`, `tests/sol_execbench/test_timing_policy.py`, or `tests/sol_execbench/test_rocm_profiler.py`.
- Docs: `docs/rocm_timing.md`, `docs/trace.md`, and internal validation docs when evidence semantics change.

**New Scoring or Reporting Feature:**
- Primary code: `src/sol_execbench/core/scoring/`, `src/sol_execbench/core/reporting.py`, or `src/sol_execbench/core/baseline.py`.
- Tests: `tests/sol_execbench/test_amd_sol_bounds.py`, `tests/sol_execbench/test_amd_native_score.py`, `tests/sol_execbench/test_baseline_comparison.py`, or a new focused test near these files.
- Scripts: Use `scripts/run_dataset.py` for dataset-level integration behavior.

**New Driver/Staging Feature:**
- Primary code: `src/sol_execbench/driver/problem_packager.py` for staging and command construction; `src/sol_execbench/driver/templates/` for scripts copied to staging.
- Tests: `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/driver/test_build_ext.py`.

**New Example:**
- Implementation: `examples/<language>/<problem>/`.
- Required files: `definition.json`, `workload.jsonl`, `reference.py` when the definition/reference is file-backed, `solution_<language>.json`, and candidate source files such as `kernel.py`, `kernel.hip`, or `main.cpp`.
- Tests: `tests/examples/test_examples.py`; add targeted tests under `tests/sol_execbench/` for new schema/runtime behavior.

**Utilities:**
- Shared benchmark helpers: `src/sol_execbench/core/bench/`.
- Shared schema helpers: `src/sol_execbench/core/data/`.
- Environment/diagnostic helpers: `src/sol_execbench/core/diagnostics.py` or `src/sol_execbench/core/utils.py`.
- Operational scripts: `scripts/`.

## Special Directories

**`src/sol_execbench/driver/templates/`:**
- Purpose: Source templates copied into staging directories.
- Generated: No; the files are source-controlled templates.
- Committed: Yes.

**`tests/sol_execbench/samples/`:**
- Purpose: Checked-in benchmark fixtures and adversarial solution samples.
- Generated: No.
- Committed: Yes.

**`examples/`:**
- Purpose: Runnable examples and compatibility fixtures.
- Generated: No.
- Committed: Yes.

**`data/`:**
- Purpose: Downloaded benchmark assets.
- Generated: Yes for contents beyond `.gitkeep`.
- Committed: Only `data/.gitkeep`.

**`.planning/`:**
- Purpose: GSD planning state, milestones, and generated codebase maps.
- Generated: Yes.
- Committed: Project-dependent planning artifacts are tracked by the GSD workflow.

**`.ruff_cache/`, `.pytest_cache/`, `.venv/`:**
- Purpose: Local tool caches and virtual environment.
- Generated: Yes.
- Committed: No.

---

*Structure analysis: 2026-05-22*
