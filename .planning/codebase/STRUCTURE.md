# Codebase Structure

**Analysis Date:** 2026-06-01

## Directory Layout

```text
SOL-ExecBench-ROCm/
├── src/sol_execbench/          # Python package source
│   ├── cli/                    # Click console entry points
│   ├── core/                   # Schemas, benchmark runtime helpers, dataset, evidence, scoring
│   │   ├── bench/              # Correctness, IO, timing, profiler, static evidence, guardrails
│   │   ├── data/               # Public Pydantic benchmark schemas
│   │   ├── dataset/            # Dataset layout, manifest, readiness, closure, sharding
│   │   └── scoring/            # AMD scoring, SOL bounds, hardware models, SOLAR derivation
│   ├── data/                   # Packaged AMD hardware model JSON
│   └── driver/                 # Staging packager and generated subprocess templates
├── tests/                      # Pytest suite
│   ├── sol_execbench/          # Package tests and sample problems
│   ├── examples/               # Example workflow tests
│   └── docker/dependencies/    # ROCm/container dependency checks
├── examples/                   # Runnable benchmark examples by solution category
├── scripts/                    # Dataset, report, Docker, and utility scripts
├── docs/                       # User, schema, architecture, ROCm, and evidence docs
├── docker/                     # ROCm container files and target manifest
├── data/                       # Downloaded benchmark assets and local runtime data
├── .github/workflows/          # CI workflows
├── .planning/                  # GSD planning and generated codebase maps
├── pyproject.toml              # Package metadata, dependencies, scripts, pytest, Ruff, Ty, uv indexes
├── uv.lock                     # Locked dependency resolution
├── README.md                   # Project overview and quick start
├── AGENTS.md                   # Repository instructions for coding agents
├── CONTRIBUTING.md             # Contribution process
├── SECURITY.md                 # Security policy
└── LICENSE                     # Apache-2.0 license
```

## Directory Purposes

**`src/sol_execbench/`:**
- Purpose: Installable Python package for SOL ExecBench ROCm.
- Contains: Public package exports, CLI, core services, packaged data, driver templates.
- Key files: `src/sol_execbench/__init__.py`, `src/sol_execbench/sol_score.py`

**`src/sol_execbench/cli/`:**
- Purpose: User-facing console commands.
- Contains: Evaluator CLI, baseline comparison CLI, CLI package export.
- Key files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/cli/baseline.py`, `src/sol_execbench/cli/__init__.py`

**`src/sol_execbench/core/data/`:**
- Purpose: Public benchmark schemas and serialization helpers.
- Contains: Pydantic models for definitions, workloads, solutions, traces, contracts, shapes, dtypes.
- Key files: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/data/contract.py`

**`src/sol_execbench/core/bench/`:**
- Purpose: Runtime logic used by the generated evaluator.
- Contains: Input generation, output normalization, correctness checks, timing, clock locking, profiler/static evidence, reward-hack checks.
- Key files: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/eval_runtime.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`

**`src/sol_execbench/core/bench/config/`:**
- Purpose: Benchmark and device clock configuration.
- Contains: Dataclass config models and clock preset data.
- Key files: `src/sol_execbench/core/bench/config/benchmark_config.py`, `src/sol_execbench/core/bench/config/device_config.py`

**`src/sol_execbench/core/dataset/`:**
- Purpose: Dataset-scale importable services.
- Contains: Category validation, checksums, layout inspection, inventory, manifest, readiness, ready subset, run state, run closure, evidence refs, sharding, parity/paper denominator helpers.
- Key files: `src/sol_execbench/core/dataset/runner.py`, `src/sol_execbench/core/dataset/run_state.py`, `src/sol_execbench/core/dataset/run_closure.py`, `src/sol_execbench/core/dataset/layout.py`, `src/sol_execbench/core/dataset/manifest.py`, `src/sol_execbench/core/dataset/sharding.py`

**`src/sol_execbench/core/scoring/`:**
- Purpose: AMD-native scoring, bound estimation, and SOLAR derivation evidence.
- Contains: AMD score artifacts, hardware models, bound graphs, estimate families, SOLAR derivation, baseline artifact handling.
- Key files: `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_sol.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/solar_derivation.py`

**`src/sol_execbench/core/`:**
- Purpose: Shared reports, evidence models, diagnostics, compatibility, dependency matrix, and public core exports.
- Contains: Environment snapshots, toolchain routing, compatibility matrix, Docker matrix, reporting, trust summary, consistency, claim upgrade, runtime evidence.
- Key files: `src/sol_execbench/core/__init__.py`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/compatibility.py`, `src/sol_execbench/core/reporting.py`, `src/sol_execbench/core/baseline.py`

**`src/sol_execbench/driver/`:**
- Purpose: Stage problem files and generated scripts for subprocess execution.
- Contains: `ProblemPackager`, compile template, evaluation template.
- Key files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/driver/templates/eval_driver.py`

**`src/sol_execbench/data/`:**
- Purpose: Static package data.
- Contains: AMD hardware model JSON used by scoring helpers.
- Key files: `src/sol_execbench/data/amd_hardware_models/gfx1200.json`

**`scripts/`:**
- Purpose: Repo-level operational commands that call package services.
- Contains: Dataset runner, downloader, dataset inspector, compatibility matrix exports, report generators, Docker wrapper.
- Key files: `scripts/run_dataset.py`, `scripts/download_solexecbench.py`, `scripts/inspect_dataset.py`, `scripts/report_amd_bound_sanity.py`, `scripts/report_trust_summary.py`, `scripts/run_docker.sh`

**`tests/`:**
- Purpose: Verification suite for package behavior, examples, and runtime environment checks.
- Contains: Unit/integration tests, sample problem fixtures, Docker dependency tests, example tests.
- Key files: `tests/conftest.py`, `tests/sol_execbench/test_e2e.py`, `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/core/bench/test_timing.py`, `tests/examples/test_examples.py`

**`examples/`:**
- Purpose: Runnable examples for supported and migration-relevant solution categories.
- Contains: Problem directories with `definition.json`, `workload.jsonl`, solution JSON, reference/source files.
- Key files: `examples/pytorch/gemma3_swiglu/solution_python.json`, `examples/triton/rmsnorm/solution_triton.json`, `examples/hip_cpp/rmsnorm/solution_hip.json`, `examples/hipblas/gemm/solution_hipblas.json`

**`docs/`:**
- Purpose: Public and internal documentation.
- Contains: Getting started, architecture, development, testing, schema docs, ROCm notes, toolchain routing, evidence quality, internal validation docs.
- Key files: `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md`, `docs/TESTING.md`, `docs/definition.md`, `docs/workload.md`, `docs/solution.md`, `docs/trace.md`

**`docker/`:**
- Purpose: ROCm container runtime support.
- Contains: Dockerfile, entrypoint, ROCm target manifest.
- Key files: `docker/Dockerfile`, `docker/entrypoint.sh`, `docker/rocm-targets.json`

**`data/`:**
- Purpose: Local downloaded benchmark assets and runtime data.
- Contains: User-managed benchmark data; excluded from Ruff checks.
- Key files: Not applicable; generated/downloaded content belongs here.

## Key File Locations

**Entry Points:**
- `pyproject.toml`: Defines console scripts `sol-execbench` and `sol-execbench-baseline`.
- `src/sol_execbench/cli/main.py`: Main evaluator and metadata subcommands.
- `src/sol_execbench/cli/baseline.py`: Baseline comparison CLI.
- `scripts/run_dataset.py`: Dataset batch runner.
- `scripts/run_docker.sh`: Docker build/entry wrapper.

**Configuration:**
- `pyproject.toml`: Dependencies, Python range, pytest markers, Ruff exclusions, Ty source roots, uv indexes.
- `uv.lock`: Locked dependency graph.
- `docker/rocm-targets.json`: ROCm Docker target metadata.
- `src/sol_execbench/core/bench/config/benchmark_config.py`: Runtime benchmark defaults.
- `src/sol_execbench/core/bench/config/device_config.py`: Device clock preset data.

**Core Logic:**
- `src/sol_execbench/core/data/definition.py`: Definition schema and axis/shape resolution.
- `src/sol_execbench/core/data/workload.py`: Workload input and tolerance schema.
- `src/sol_execbench/core/data/solution.py`: Solution schema, ROCm language validation, compile flag guardrails.
- `src/sol_execbench/core/data/trace.py`: Canonical evaluation trace schema.
- `src/sol_execbench/driver/problem_packager.py`: Staging lifecycle and native compile command generation.
- `src/sol_execbench/driver/templates/eval_driver.py`: Per-workload evaluation loop.
- `src/sol_execbench/core/bench/eval_runtime.py`: Import, timing, trace emission, and reward-hack helper functions for the driver.
- `src/sol_execbench/core/bench/timing.py`: HIP-backed PyTorch device-event timing path.
- `src/sol_execbench/core/dataset/runner.py`: Importable dataset-run helpers.
- `src/sol_execbench/core/scoring/amd_score.py`: AMD-native score report construction.

**Testing:**
- `tests/conftest.py`: Pytest marker/environment behavior.
- `tests/sol_execbench/core/data/`: Schema unit tests.
- `tests/sol_execbench/core/bench/`: Benchmark helper tests.
- `tests/sol_execbench/driver/`: Staging, compile template, and eval-driver tests.
- `tests/sol_execbench/samples/`: Sample problem fixtures.
- `tests/examples/`: Example coverage.
- `tests/docker/dependencies/`: ROCm/Docker dependency checks.

**Documentation:**
- `README.md`: Project overview and quick start.
- `docs/ARCHITECTURE.md`: User-facing architecture narrative.
- `docs/DEVELOPMENT.md`: Development workflow and source areas.
- `docs/TESTING.md`: Test commands and marker guidance.
- `docs/CONFIGURATION.md`: CLI and environment configuration.
- `docs/rocm_timing.md`: Timing methodology.
- `docs/static_kernel_evidence.md`: Static evidence sidecar behavior.

## Naming Conventions

**Files:**
- Python modules use `snake_case.py`: `src/sol_execbench/core/dataset/run_closure.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`.
- Tests use `test_*.py`: `tests/sol_execbench/test_dataset_runner.py`, `tests/sol_execbench/core/bench/test_timing.py`.
- Example problem files use fixed benchmark names: `definition.json`, `workload.jsonl`, `solution_*.json`, `reference.py`, `kernel.py`, `kernel.hip`, `main.cpp`.
- Generated/evidence reports use descriptive JSON suffixes: `*.timing.json`, `*.amd-sol-v2.json`, `*.solar-derivation.json`, `*.static-evidence.json`.

**Directories:**
- Package subpackages are lowercase nouns: `cli`, `core`, `driver`, `data`.
- Core feature subpackages are domain nouns: `bench`, `dataset`, `scoring`.
- Examples are grouped by solution category: `examples/pytorch/`, `examples/triton/`, `examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/ck/`, `examples/rocwmma/`.
- Tests mirror package or workflow ownership: `tests/sol_execbench/core/bench/`, `tests/sol_execbench/driver/`, `tests/examples/`, `tests/docker/dependencies/`.

## Where to Add New Code

**New CLI flag or evaluator behavior:**
- Primary code: `src/sol_execbench/cli/main.py`
- Runtime helper: `src/sol_execbench/core/bench/` when behavior must run inside the generated evaluator.
- Tests: `tests/sol_execbench/test_e2e.py`, `tests/sol_execbench/driver/test_eval_driver.py`, or a focused `tests/sol_execbench/test_<feature>.py`.

**New benchmark schema field or validation rule:**
- Primary code: `src/sol_execbench/core/data/`
- Contract docs: `docs/definition.md`, `docs/workload.md`, `docs/solution.md`, or `docs/trace.md`
- Tests: `tests/sol_execbench/core/data/`

**New generated evaluation behavior:**
- Driver template: `src/sol_execbench/driver/templates/eval_driver.py`
- Importable helper: `src/sol_execbench/core/bench/eval_runtime.py` or another module under `src/sol_execbench/core/bench/`
- Tests: `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/core/bench/test_eval_runtime.py`

**New native ROCm compile behavior:**
- Primary code: `src/sol_execbench/driver/problem_packager.py`
- Template changes: `src/sol_execbench/driver/templates/build_ext.py`
- Schema validation: `src/sol_execbench/core/data/solution.py`
- Tests: `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/driver/test_build_ext.py`

**New dataset workflow:**
- Reusable logic: `src/sol_execbench/core/dataset/`
- Command-line glue: `scripts/run_dataset.py` or a focused script in `scripts/`
- Tests: `tests/sol_execbench/test_dataset_*.py` and `tests/sol_execbench/test_run_dataset_*.py`

**New evidence sidecar:**
- Primary model/helper: `src/sol_execbench/core/` for global evidence, `src/sol_execbench/core/bench/` for runtime evidence, or `src/sol_execbench/core/dataset/` for dataset-run evidence.
- CLI writer: `src/sol_execbench/cli/main.py` for per-evaluation sidecars or `scripts/run_dataset.py` for dataset sidecars.
- Tests: Focused `tests/sol_execbench/test_*evidence*.py` or existing evidence test files.

**New scoring or bound model:**
- Primary code: `src/sol_execbench/core/scoring/`
- Static data: `src/sol_execbench/data/amd_hardware_models/`
- Tests: `tests/sol_execbench/test_amd_*.py`, `tests/sol_execbench/test_solar_derivation_*.py`, or a new focused scoring test.

**New documentation:**
- User-facing docs: `docs/`
- Internal validation notes: `docs/internal/`
- Codebase map artifacts: `.planning/codebase/`

**New example problem:**
- Implementation: `examples/<category>/<problem_name>/`
- Required files: `definition.json`, `workload.jsonl`, `solution_*.json`, source file(s), and `reference.py` when not fully inlined.
- Tests: `tests/examples/test_examples.py` and, when category-specific, `tests/examples/test_rocm_cli_paths.py`.

**Utilities:**
- Shared benchmark helpers: `src/sol_execbench/core/bench/`
- Shared dataset helpers: `src/sol_execbench/core/dataset/`
- Shared report/scoring helpers: `src/sol_execbench/core/` or `src/sol_execbench/core/scoring/`
- Repo-only command glue: `scripts/`

## Special Directories

**`.planning/`:**
- Purpose: GSD project plans, milestones, notes, generated codebase maps.
- Generated: Yes
- Committed: Project-dependent; codebase maps under `.planning/codebase/` are intended artifacts for GSD workflows.

**`.uv-cache/`:**
- Purpose: Local uv cache.
- Generated: Yes
- Committed: No

**`.venv/`:**
- Purpose: Local virtual environment.
- Generated: Yes
- Committed: No

**`.ruff_cache/`:**
- Purpose: Ruff cache.
- Generated: Yes
- Committed: No

**`.pytest_cache/`:**
- Purpose: Pytest cache.
- Generated: Yes
- Committed: No

**`dist/`:**
- Purpose: Built package artifacts.
- Generated: Yes
- Committed: No

**`.artifacts/`:**
- Purpose: Local validation/evidence artifacts.
- Generated: Yes
- Committed: No unless explicitly curated.

**`data/`:**
- Purpose: Downloaded benchmark assets and local runtime data.
- Generated: Yes
- Committed: No for downloaded datasets.

**`examples/`:**
- Purpose: Source-controlled sample problem directories.
- Generated: No
- Committed: Yes

**`src/sol_execbench/driver/templates/`:**
- Purpose: Versioned Python scripts copied into staging directories at runtime.
- Generated: No
- Committed: Yes

**`tests/sol_execbench/samples/`:**
- Purpose: Source-controlled fixture problem directories and adversarial sample solutions.
- Generated: No
- Committed: Yes

---

*Structure analysis: 2026-06-01*
