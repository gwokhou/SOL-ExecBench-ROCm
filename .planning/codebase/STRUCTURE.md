# Codebase Structure

**Analysis Date:** 2026-05-31

## Directory Layout

```text
SOL-ExecBench-ROCm/
+-- src/sol_execbench/              # Python package source
|   +-- cli/                        # Click command entry points
|   +-- core/                       # Benchmark schemas, execution helpers, scoring, reports
|   |   +-- bench/                  # GPU execution primitives and evidence collection
|   |   |   +-- config/             # Benchmark and device/clock config dataclasses
|   |   +-- data/                   # Pydantic public contract models and JSON helpers
|   |   +-- dataset/                # Dataset layout, manifest, readiness, parity, closure reports
|   |   +-- scoring/                # AMD SOL bounds, hardware models, derived scores
|   +-- data/                       # Packaged static data such as AMD hardware model JSON
|   +-- driver/                     # Staging packager and copied subprocess templates
+-- tests/                          # Pytest suite
|   +-- sol_execbench/              # Package unit/integration tests and sample problems
|   +-- examples/                   # Example workflow tests
|   +-- docker/                     # Container dependency/runtime tests
+-- scripts/                        # Dataset, report, Docker, and utility command scripts
+-- examples/                       # Runnable solution examples by language/library category
+-- docs/                           # User, developer, schema, ROCm, and release documentation
+-- docker/                         # ROCm container image, entrypoint, target config
+-- data/                           # Downloaded benchmark assets; only `.gitkeep` is committed
+-- .planning/                      # GSD project state and generated planning/codebase docs
+-- pyproject.toml                  # Package metadata, scripts, dependencies, tool config
+-- uv.lock                         # Locked dependency resolution
+-- AGENTS.md                       # Repository instructions for coding agents
+-- README.md                       # Project overview and usage
+-- SECURITY.md                     # Security policy
```

## Directory Purposes

**`src/sol_execbench/`:**
- Purpose: Installable Python package.
- Contains: Public package exports, CLI package, core domain modules, packaged data, driver staging code.
- Key files: `src/sol_execbench/__init__.py`, `src/sol_execbench/sol_score.py`.

**`src/sol_execbench/cli/`:**
- Purpose: User-facing console commands.
- Contains: Root evaluator CLI and baseline comparison CLI.
- Key files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/cli/baseline.py`, `src/sol_execbench/cli/__init__.py`.

**`src/sol_execbench/core/`:**
- Purpose: Core benchmark logic independent of command-line parsing.
- Contains: Public re-export module, baseline comparison, compatibility matrices, environment/toolchain diagnostics, runtime evidence, reporting, dataset/scoring/bench/data packages.
- Key files: `src/sol_execbench/core/__init__.py`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/reporting.py`.

**`src/sol_execbench/core/data/`:**
- Purpose: Schema-first public contract for benchmark inputs and outputs.
- Contains: Pydantic models for definitions, workloads, solutions, traces, evaluator contract, dtypes, shapes, JSON utilities.
- Key files: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/data/contract.py`.

**`src/sol_execbench/core/bench/`:**
- Purpose: GPU evaluation helpers used by the generated evaluation driver and CLI sidecars.
- Contains: Input/output generation, correctness checks, timing, reward-hack defenses, clock locking, profiler integration, static kernel evidence.
- Key files: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`.

**`src/sol_execbench/core/bench/config/`:**
- Purpose: Runtime benchmark and ROCm device/clock configuration.
- Contains: Dataclasses and clock preset lookup.
- Key files: `src/sol_execbench/core/bench/config/benchmark_config.py`, `src/sol_execbench/core/bench/config/device_config.py`.

**`src/sol_execbench/core/dataset/`:**
- Purpose: Dataset inspection and release/readiness metadata.
- Contains: Category validation, checksums, layout inspection, manifest generation, inventory, readiness, ready subsets, parity gaps, paper denominator, execution closure.
- Key files: `src/sol_execbench/core/dataset/layout.py`, `src/sol_execbench/core/dataset/inventory.py`, `src/sol_execbench/core/dataset/readiness.py`, `src/sol_execbench/core/dataset/execution_closure.py`, `src/sol_execbench/core/dataset/manifest.py`.

**`src/sol_execbench/core/scoring/`:**
- Purpose: AMD-native derived score and SOL-bound logic.
- Contains: AMD hardware model loading, bound graph and estimates, SOL artifacts, scoring baseline artifacts, guarded score reports, solar derivation.
- Key files: `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_sol.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/amd_hardware_models.py`, `src/sol_execbench/core/scoring/baseline_artifact.py`.

**`src/sol_execbench/data/`:**
- Purpose: Static data bundled with the package.
- Contains: AMD hardware model JSON.
- Key files: `src/sol_execbench/data/amd_hardware_models/gfx1200.json`.

**`src/sol_execbench/driver/`:**
- Purpose: Evaluation staging and generated subprocess templates.
- Contains: `ProblemPackager`, template scripts copied into staging directories.
- Key files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/templates/build_ext.py`.

**`tests/`:**
- Purpose: Pytest coverage for schemas, CLI behavior, driver templates, ROCm behavior, examples, Docker dependency checks, docs/report contracts.
- Contains: `tests/sol_execbench/` package tests, `tests/examples/`, `tests/docker/`, fixtures and sample problem data.
- Key files: `tests/conftest.py`, `tests/sol_execbench/test_e2e.py`, `tests/sol_execbench/driver/test_problem_packager.py`, `tests/examples/test_examples.py`.

**`scripts/`:**
- Purpose: Operational utilities outside the importable package.
- Contains: Dataset downloader/runner/inspector, report generators, matrix utilities, Docker wrapper.
- Key files: `scripts/run_dataset.py`, `scripts/download_solexecbench.py`, `scripts/inspect_dataset.py`, `scripts/run_docker.sh`.

**`examples/`:**
- Purpose: Runnable solution examples grouped by implementation category.
- Contains: ROCm-native examples (`hip_cpp`, `hipblas`, `miopen`, `ck`, `rocwmma`), PyTorch/Triton examples, and legacy CUDA/NVIDIA category examples used for migration/parity context.
- Key files: Example problem directories under `examples/hip_cpp/`, `examples/triton/`, `examples/pytorch/`, `examples/hipblas/`.

**`docs/`:**
- Purpose: Human documentation for usage, schemas, ROCm migration, testing, evidence, and release closure.
- Contains: Public docs and internal validation handoffs.
- Key files: `docs/ARCHITECTURE.md`, `docs/GETTING-STARTED.md`, `docs/DEVELOPMENT.md`, `docs/TESTING.md`, `docs/definition.md`, `docs/solution.md`, `docs/trace.md`, `docs/rocm.md`.

**`docker/`:**
- Purpose: ROCm-capable evaluation environment.
- Contains: Container image, entrypoint, target architecture metadata.
- Key files: `docker/Dockerfile`, `docker/entrypoint.sh`, `docker/rocm-targets.json`.

**`data/`:**
- Purpose: Local downloaded benchmark assets.
- Contains: `.gitkeep` only in source control; downloaded benchmark directories belong here.
- Key files: `data/.gitkeep`.

## Key File Locations

**Entry Points:**
- `pyproject.toml`: Declares console scripts `sol-execbench` and `sol-execbench-baseline`.
- `src/sol_execbench/cli/main.py`: Root evaluator CLI and metadata subcommands.
- `src/sol_execbench/cli/baseline.py`: Trace baseline comparison CLI.
- `scripts/run_dataset.py`: Batch runner for single problems and dataset category roots.
- `scripts/run_docker.sh`: Docker build/run wrapper.

**Configuration:**
- `pyproject.toml`: Python version, dependencies, console scripts, pytest markers, Ruff/Ty/uv config.
- `.python-version`: Local Python version selector.
- `.pre-commit-config.yaml`: Pre-commit hook configuration.
- `docker/rocm-targets.json`: Docker/ROCm target metadata.
- `src/sol_execbench/core/bench/config/benchmark_config.py`: Runtime benchmark defaults.
- `src/sol_execbench/core/bench/config/device_config.py`: ROCm clock preset configuration.

**Core Logic:**
- `src/sol_execbench/core/data/definition.py`: Definition schema and shape/axis resolution.
- `src/sol_execbench/core/data/workload.py`: Workload schema and input/tolerance specs.
- `src/sol_execbench/core/data/solution.py`: Solution/build schema and ROCm language validation.
- `src/sol_execbench/core/data/trace.py`: Canonical trace/evaluation output schema.
- `src/sol_execbench/core/bench/io.py`: Input generation, safetensors loading, output allocation.
- `src/sol_execbench/core/bench/correctness.py`: Numerical correctness and reproducible seed handling.
- `src/sol_execbench/core/bench/timing.py`: HIP-backed PyTorch device-event timing.
- `src/sol_execbench/core/bench/reward_hack.py`: Static and runtime reward-hack detection.
- `src/sol_execbench/driver/problem_packager.py`: Staging and command generation.
- `src/sol_execbench/driver/templates/eval_driver.py`: GPU-side evaluation process.

**Testing:**
- `tests/sol_execbench/`: Primary package tests.
- `tests/sol_execbench/driver/`: Driver template and staging tests.
- `tests/sol_execbench/samples/`: Sample problem fixtures used by tests.
- `tests/examples/`: Example workflow tests.
- `tests/docker/dependencies/`: Container and ROCm dependency tests.

**Documentation:**
- `README.md`: Project entry documentation.
- `docs/`: Maintained public docs.
- `.planning/codebase/`: Generated codebase maps consumed by GSD planning/execution commands.

## Naming Conventions

**Files:**
- Python source modules use `snake_case.py`: `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/dataset/paper_denominator.py`.
- Test files use `test_*.py`: `tests/sol_execbench/test_contract.py`, `tests/examples/test_examples.py`.
- Scripts use command-style `snake_case.py` or shell names: `scripts/run_dataset.py`, `scripts/run_docker.sh`.
- Public docs use descriptive Markdown names: `docs/GETTING-STARTED.md`, `docs/static_kernel_evidence.md`.
- Generated codebase docs use uppercase Markdown names: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`.

**Directories:**
- Package directories use lowercase snake_case where needed: `src/sol_execbench/`, `src/sol_execbench/core/`.
- Benchmark example directories are grouped by implementation category: `examples/hip_cpp/`, `examples/triton/`, `examples/miopen/`.
- Test sample directories mirror problem or exploit names: `tests/sol_execbench/samples/rmsnorm/`, `tests/sol_execbench/samples/evil_monkey_patch/`.

## Where to Add New Code

**New CLI Command:**
- Primary code: add a Click command in `src/sol_execbench/cli/main.py` if it belongs under `sol-execbench`, or a sibling module in `src/sol_execbench/cli/` for a separate console script.
- Tests: add command tests under `tests/sol_execbench/test_*cli*.py`.
- Packaging: update `[project.scripts]` in `pyproject.toml` only for new top-level commands.

**New Schema Field or Contract Model:**
- Primary code: add Pydantic model logic under `src/sol_execbench/core/data/`.
- Tests: add model and public contract tests under `tests/sol_execbench/test_contract.py` or a focused `tests/sol_execbench/test_<schema>.py`.
- Docs: update schema docs under `docs/definition.md`, `docs/workload.md`, `docs/solution.md`, or `docs/trace.md`.

**New Benchmark Execution Behavior:**
- Primary code: put GPU execution primitives in `src/sol_execbench/core/bench/`.
- Subprocess integration: wire execution behavior into `src/sol_execbench/driver/templates/eval_driver.py`.
- Parent CLI integration: keep orchestration in `src/sol_execbench/cli/main.py`.
- Tests: add unit tests under `tests/sol_execbench/core/bench/` or integration tests under `tests/sol_execbench/test_e2e.py`.

**New Native ROCm Language or Library Category:**
- Schema: update `SupportedLanguages` and validation in `src/sol_execbench/core/data/solution.py`.
- Staging/compile: update `_CPP_LANGUAGES` and build handling in `src/sol_execbench/driver/problem_packager.py`.
- Evaluation import: update native language handling in `src/sol_execbench/driver/templates/eval_driver.py`.
- Examples: add examples under `examples/<category>/`.
- Tests: add schema, packager, and example tests under `tests/sol_execbench/` and `tests/examples/`.

**New Dataset Report:**
- Primary code: add reusable report logic in `src/sol_execbench/core/dataset/`.
- Script wrapper: add a command script in `scripts/report_<name>.py`.
- Tests: add focused report tests under `tests/sol_execbench/test_<report>.py`.
- Docs: add or update report documentation under `docs/`.

**New Scoring or Bound Logic:**
- Primary code: add scoring implementation under `src/sol_execbench/core/scoring/`.
- Static data: add hardware model data under `src/sol_execbench/data/amd_hardware_models/` when it is package data.
- Tests: add scoring tests under `tests/sol_execbench/test_amd_*.py` or focused fixture-backed tests under `tests/sol_execbench/fixtures/`.

**Utilities:**
- Shared importable helpers: use the closest domain package under `src/sol_execbench/core/`.
- One-off operational tools: use `scripts/`.
- Test-only helpers: use `tests/sol_execbench_type_helpers.py` or a focused helper next to tests.

## Special Directories

**`.planning/`:**
- Purpose: GSD workflow state, project plans, generated codebase maps, milestone artifacts.
- Generated: Yes.
- Committed: Yes for planning artifacts relevant to project workflow.

**`.artifacts/`:**
- Purpose: Local validation/evidence artifacts from ROCm and dataset runs.
- Generated: Yes.
- Committed: No by default unless explicitly selected as evidence.

**`data/`:**
- Purpose: Downloaded SOL ExecBench benchmark assets.
- Generated: Yes after download.
- Committed: No, except `data/.gitkeep`.

**`dist/`:**
- Purpose: Python build outputs.
- Generated: Yes.
- Committed: No by default; existing wheel/sdist files are build artifacts.

**`src/sol_execbench/driver/templates/`:**
- Purpose: Source templates copied into staging directories and executed as standalone scripts.
- Generated: No.
- Committed: Yes.

**`src/sol_execbench/data/amd_hardware_models/`:**
- Purpose: Package data consumed by scoring/hardware model code.
- Generated: No.
- Committed: Yes.

**`examples/`:**
- Purpose: Runnable examples and migration category references.
- Generated: No.
- Committed: Yes.

**`tests/sol_execbench/samples/`:**
- Purpose: Test fixture problem directories and exploit samples.
- Generated: No.
- Committed: Yes.

**`.venv/`, `.uv-cache/`, `.ruff_cache/`, `.pytest_cache/`, `__pycache__/`:**
- Purpose: Local dependency, lint, pytest, and bytecode caches.
- Generated: Yes.
- Committed: No.

---

*Structure analysis: 2026-05-31*
