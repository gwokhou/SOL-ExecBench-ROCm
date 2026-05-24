# Codebase Structure

**Analysis Date:** 2026-05-24

## Directory Layout

```text
SOL-ExecBench-ROCm/
├── src/sol_execbench/              # Installable Python package
│   ├── cli/                        # Click command entry points
│   ├── core/                       # Data models, benchmark runtime, dataset helpers, scoring
│   │   ├── bench/                  # Runtime input generation, timing, correctness, profiler helpers
│   │   ├── data/                   # Pydantic public schemas and shape/dtype helpers
│   │   ├── dataset/                # Dataset manifest, inventory, readiness, parity sidecars
│   │   └── scoring/                # AMD SOL bounds, hardware models, native score reports
│   ├── data/amd_hardware_models/   # Packaged hardware model JSON
│   └── driver/                     # Problem packager and generated subprocess templates
├── tests/                          # Pytest suite and sample problem fixtures
│   ├── sol_execbench/              # Package tests
│   ├── examples/                   # Example validation tests
│   ├── docker/                     # Docker dependency tests
│   └── samples/                    # Lightweight fixture problems
├── examples/                       # Runnable solution/problem examples by backend category
├── scripts/                        # Dataset download, inspection, batch execution, reporting
├── docs/                           # User, architecture, ROCm, timing, schema, and release docs
├── docker/                         # GPU evaluation container files
├── data/                           # Downloaded benchmark assets and local dataset roots
├── .planning/                      # GSD planning and codebase map artifacts
├── pyproject.toml                  # Package metadata, dependencies, pytest markers, Ruff config
└── uv.lock                         # Locked uv dependency graph
```

## Directory Purposes

**`src/sol_execbench/`:**
- Purpose: Installable Python source package.
- Contains: Package exports, CLI modules, core library code, packaged hardware model data, and driver templates.
- Key files: `src/sol_execbench/__init__.py`, `src/sol_execbench/core/__init__.py`, `src/sol_execbench/sol_score.py`

**`src/sol_execbench/cli/`:**
- Purpose: Console command implementations.
- Contains: Main evaluator command and baseline comparison command.
- Key files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/cli/baseline.py`, `src/sol_execbench/cli/__init__.py`

**`src/sol_execbench/core/data/`:**
- Purpose: Public schema layer for benchmark inputs and outputs.
- Contains: Pydantic models for definitions, solutions, workloads, traces, evaluator contracts, dtype mapping, JSON loading, and shape expression resolution.
- Key files: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/data/contract.py`

**`src/sol_execbench/core/bench/`:**
- Purpose: Runtime benchmark mechanics used by generated evaluation scripts.
- Contains: Input/output generation, safetensors loading, correctness metrics, timing, reward-hack defenses, clock-lock checks, profiler helpers, and benchmark config.
- Key files: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/config/benchmark_config.py`

**`src/sol_execbench/core/dataset/`:**
- Purpose: Deterministic dataset layout and readiness sidecar generation.
- Contains: Category validation, layout inspection, checksums, manifest, inventory, readiness, ready-subset, and parity-gap report builders.
- Key files: `src/sol_execbench/core/dataset/manifest.py`, `src/sol_execbench/core/dataset/inventory.py`, `src/sol_execbench/core/dataset/readiness.py`, `src/sol_execbench/core/dataset/ready_subset.py`, `src/sol_execbench/core/dataset/parity_gap.py`

**`src/sol_execbench/core/scoring/`:**
- Purpose: Derived AMD-native scoring and speed-of-light evidence.
- Contains: AMD hardware models, bound graph construction, work estimates, SOL bound artifacts, native score reports, baseline artifacts, and solar derivation evidence.
- Key files: `src/sol_execbench/core/scoring/amd_sol.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_hardware_models.py`

**`src/sol_execbench/data/`:**
- Purpose: Packaged static data used by the library.
- Contains: AMD hardware model JSON files.
- Key files: `src/sol_execbench/data/amd_hardware_models/gfx1200.json`

**`src/sol_execbench/driver/`:**
- Purpose: Boundary between validated model objects and subprocess-isolated execution.
- Contains: `ProblemPackager` and self-contained Python templates copied into staging directories.
- Key files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/driver/templates/eval_driver.py`

**`scripts/`:**
- Purpose: Developer and release workflow commands that are not packaged console scripts.
- Contains: Dataset download, dataset inspection, dataset batch execution, parity-gap reporting, and Docker wrapper.
- Key files: `scripts/run_dataset.py`, `scripts/download_solexecbench.py`, `scripts/inspect_dataset.py`, `scripts/report_parity_gaps.py`, `scripts/run_docker.sh`

**`examples/`:**
- Purpose: Runnable sample problems and solutions across Python, Triton, HIP/C++, and ROCm library categories.
- Contains: Per-example `definition.json`, `workload.jsonl`, `solution_*.json`, source files, and references.
- Key files: `examples/pytorch/gemma3_swiglu/solution_python.json`, `examples/triton/rmsnorm/solution_triton.json`, `examples/hip_cpp/rmsnorm/solution_hip.json`, `examples/hipblas/gemm/solution_hipblas.json`

**`tests/`:**
- Purpose: Pytest coverage for schemas, runtime logic, drivers, examples, dataset reports, scoring, docs, and ROCm migration guardrails.
- Contains: Unit tests, integration-style tests, sample problem fixtures, reward-hack samples, Docker dependency tests, and example validation.
- Key files: `tests/sol_execbench/core/data/test_definition.py`, `tests/sol_execbench/core/bench/test_timing.py`, `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/test_e2e.py`, `tests/examples/test_examples.py`

**`docs/`:**
- Purpose: User and internal documentation.
- Contains: Architecture, configuration, getting started, testing, schema docs, ROCm migration notes, timing docs, compliance docs, and internal validation notes.
- Key files: `docs/ARCHITECTURE.md`, `docs/GETTING-STARTED.md`, `docs/CONFIGURATION.md`, `docs/TESTING.md`, `docs/rocm.md`, `docs/rocm_timing.md`

**`docker/`:**
- Purpose: Containerized ROCm evaluation environment.
- Contains: Dockerfile and entrypoint.
- Key files: `docker/Dockerfile`, `docker/entrypoint.sh`

**`data/`:**
- Purpose: Local downloaded benchmark data and generated dataset assets.
- Contains: Dataset roots and benchmark artifacts. Treat this as local/generated data, not source code.
- Key files: `data/` directory placeholder and downloaded files.

**`.planning/`:**
- Purpose: GSD workflow state, plans, maps, milestones, and notes.
- Contains: Codebase maps, milestone phase folders, quick tasks, research notes, and state.
- Key files: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`

## Key File Locations

**Entry Points:**
- `src/sol_execbench/cli/main.py`: Main `sol-execbench` evaluator and `contract` command dispatcher.
- `src/sol_execbench/cli/baseline.py`: `sol-execbench-baseline` command.
- `scripts/run_dataset.py`: Batch runner for single problems or dataset roots.
- `scripts/download_solexecbench.py`: Dataset acquisition/materialization script.
- `scripts/inspect_dataset.py`: Manifest/inventory/readiness sidecar script.
- `scripts/report_parity_gaps.py`: Parity-gap report script.
- `src/sol_execbench/driver/templates/eval_driver.py`: Generated staging entry point for actual workload evaluation.
- `src/sol_execbench/driver/templates/build_ext.py`: Generated staging entry point for native HIP/C++ compilation.

**Configuration:**
- `pyproject.toml`: Build backend, package metadata, dependencies, console scripts, pytest markers, Ruff exclusions, uv package indexes.
- `uv.lock`: Locked dependency versions.
- `docker/Dockerfile`: Container build definition.
- `docker/entrypoint.sh`: Container startup behavior.
- `src/sol_execbench/core/bench/config/benchmark_config.py`: Benchmark runtime config model and presets.
- `src/sol_execbench/core/bench/config/device_config.py`: Device/clock configuration helpers.

**Core Logic:**
- `src/sol_execbench/core/data/definition.py`: Definition schema, reference AST validation, shape/axis resolution.
- `src/sol_execbench/core/data/solution.py`: Solution schema, ROCm language support, source validation, content hash.
- `src/sol_execbench/core/data/workload.py`: Workload schema, input spec variants, tolerance model.
- `src/sol_execbench/core/data/trace.py`: Trace, Evaluation, Correctness, Performance, Environment, and status schemas.
- `src/sol_execbench/core/bench/io.py`: Input generation, safetensors loading, output normalization/allocation, memory pool allocator.
- `src/sol_execbench/core/bench/timing.py`: ROCm-compatible PyTorch event timing.
- `src/sol_execbench/core/bench/correctness.py`: Numerical correctness and tolerance checks.
- `src/sol_execbench/core/bench/reward_hack.py`: Static and runtime reward-hack defenses.
- `src/sol_execbench/driver/problem_packager.py`: Staging, ROCm offload target injection, command creation, trace stdout parsing.
- `src/sol_execbench/core/scoring/amd_score.py`: Guarded AMD-native scoring.
- `src/sol_execbench/core/scoring/amd_sol.py`: AMD speed-of-light bound artifact generation.
- `src/sol_execbench/core/reporting.py`: Trace summaries and derived evidence reports.

**Testing:**
- `tests/sol_execbench/core/data/`: Schema and dtype tests.
- `tests/sol_execbench/core/bench/`: Runtime input/timing/correctness/reward-hack tests.
- `tests/sol_execbench/driver/`: Packager and generated driver tests.
- `tests/sol_execbench/samples/`: Sample problem fixtures for evaluator tests.
- `tests/sol_execbench/fixtures/solar_derivation/`: JSON fixtures for solar derivation tests.
- `tests/examples/test_examples.py`: Example problem coverage.
- `tests/docker/`: Docker/dependency validation tests.

**Documentation:**
- `README.md`: Project overview and basic usage.
- `AGENTS.md`: Repository instructions for coding agents.
- `CONTRIBUTING.md`: Contribution, commit, and PR expectations.
- `SECURITY.md`: Security reporting guidance.
- `docs/`: Detailed user, architecture, configuration, testing, and ROCm documentation.

## Naming Conventions

**Files:**
- Python package modules use `snake_case.py`: `problem_packager.py`, `amd_score.py`, `solar_derivation.py`.
- Test files use `test_*.py`: `tests/sol_execbench/test_public_contract_guardrails.py`.
- JSON examples use benchmark contract names: `definition.json`, `workload.jsonl`, `solution_*.json`, `config.json`.
- Generated sidecar files use explicit schema/report suffixes in scripts, such as `.timing.json`, `.solar-derivation.json`, `.amd-sol-v2.json`.
- Docs use uppercase names for primary guides and lowercase topical files: `docs/ARCHITECTURE.md`, `docs/rocm_timing.md`.

**Directories:**
- Package directories are lowercase and domain-oriented: `core/data`, `core/bench`, `core/dataset`, `core/scoring`.
- Example directories are grouped by backend category, then problem name: `examples/triton/rmsnorm`, `examples/hip_cpp/rmsnorm`, `examples/hipblas/gemm`.
- Test directories mirror package layers where practical: `tests/sol_execbench/core/data`, `tests/sol_execbench/core/bench`, `tests/sol_execbench/driver`.
- Sample fixture directories use descriptive problem or behavior names: `tests/sol_execbench/samples/evil_monkey_patch`.

## Where to Add New Code

**New Schema Field Or Contract Rule:**
- Primary code: `src/sol_execbench/core/data/`
- Tests: `tests/sol_execbench/core/data/` and public guardrails in `tests/sol_execbench/test_public_contract_guardrails.py`
- Docs/examples: Update matching `docs/*.md` and relevant `examples/*/*/*.json`

**New Evaluator CLI Option:**
- Primary code: `src/sol_execbench/cli/main.py`
- Runtime support: `src/sol_execbench/core/bench/config/` if it changes benchmark behavior.
- Tests: `tests/sol_execbench/test_e2e.py`, `tests/sol_execbench/test_public_contract_guardrails.py`, or targeted CLI tests.

**New Benchmark Runtime Behavior:**
- Primary code: `src/sol_execbench/core/bench/`
- Generated driver integration: `src/sol_execbench/driver/templates/eval_driver.py`
- Tests: `tests/sol_execbench/core/bench/` and driver/e2e tests when subprocess behavior changes.

**New Native ROCm Language Or Library Category:**
- Schema: `src/sol_execbench/core/data/solution.py`
- Staging classification: `src/sol_execbench/driver/problem_packager.py`
- Eval import dispatch: `src/sol_execbench/driver/templates/eval_driver.py`
- Examples: `examples/<category>/<problem>/`
- Tests: `tests/sol_execbench/test_rocm_library_examples.py`, `tests/examples/test_examples.py`, and schema tests.

**New Dataset Sidecar Or Readiness Report:**
- Primary code: `src/sol_execbench/core/dataset/`
- Script integration: `scripts/inspect_dataset.py`, `scripts/run_dataset.py`, or `scripts/report_parity_gaps.py`
- Tests: `tests/sol_execbench/test_dataset_contract.py`, `tests/sol_execbench/test_dataset_inventory_readiness.py`, or a new targeted `tests/sol_execbench/test_<sidecar>.py`

**New AMD Scoring Or Bound Logic:**
- Primary code: `src/sol_execbench/core/scoring/`
- Hardware data: `src/sol_execbench/data/amd_hardware_models/` if model constants change.
- Dataset runner integration: `scripts/run_dataset.py`
- Tests: `tests/sol_execbench/test_amd_*.py`, `tests/sol_execbench/test_solar_derivation_*.py`

**New Reporting Helper:**
- Primary code: `src/sol_execbench/core/reporting.py` for trace-derived reports, or `src/sol_execbench/core/scoring/` for score-specific reports.
- Tests: `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py`

**New Script:**
- Implementation: `scripts/`
- Tests: `tests/sol_execbench/` using `importlib.util` patterns seen in existing script tests.
- Docs: `docs/` and `README.md` when user-facing.

**New Example Problem:**
- Implementation: `examples/<backend>/<problem_name>/`
- Required files: `definition.json`, `workload.jsonl`, `solution_*.json`, source files referenced by the solution JSON.
- Tests: `tests/examples/test_examples.py` and backend-specific ROCm example tests.

**Utilities:**
- Shared package helpers: `src/sol_execbench/core/utils.py`
- Data parsing helpers: `src/sol_execbench/core/data/json_utils.py`
- Dataset checksum/category helpers: `src/sol_execbench/core/dataset/checksums.py`, `src/sol_execbench/core/dataset/categories.py`
- Avoid adding broad utility modules when a helper only serves one layer.

## Special Directories

**`src/sol_execbench/driver/templates/`:**
- Purpose: Source templates copied into per-run staging directories.
- Generated: No, templates are committed source files.
- Committed: Yes.
- Guidance: Keep these scripts self-contained and careful about stdout because they run in subprocess staging directories.

**`examples/`:**
- Purpose: Runnable examples and contract fixtures.
- Generated: No, but excluded from Ruff checks in `pyproject.toml`.
- Committed: Yes.
- Guidance: Keep example JSON and source files valid against the current schemas.

**`tests/sol_execbench/samples/`:**
- Purpose: Small benchmark fixtures and adversarial samples used by tests.
- Generated: No.
- Committed: Yes.
- Guidance: Add minimal fixtures here when testing driver/runtime behavior.

**`tests/sol_execbench/fixtures/`:**
- Purpose: Structured JSON fixtures for scoring/derivation tests.
- Generated: No.
- Committed: Yes.
- Guidance: Prefer fixed JSON fixtures for deterministic evidence tests.

**`data/`:**
- Purpose: Downloaded benchmark assets and local data roots.
- Generated: Yes.
- Committed: Directory is present, downloaded contents should not be committed.
- Guidance: Scripts may read/write here; do not store secrets or generated benchmark output in source commits.

**`dist/`:**
- Purpose: Built Python package distributions.
- Generated: Yes.
- Committed: Present in working tree, should generally be treated as build output.

**`.planning/`:**
- Purpose: GSD workflow artifacts, codebase maps, phases, and notes.
- Generated: Yes, by GSD workflows.
- Committed: Managed by workflow.
- Guidance: Codebase mappers should write only assigned files under `.planning/codebase/`.

**`.uv-cache/`, `.ruff_cache/`, `.pytest_cache/`, `.venv/`:**
- Purpose: Local tool caches and virtual environment.
- Generated: Yes.
- Committed: No.
- Guidance: Do not inspect or modify unless diagnosing environment/tooling behavior.

---

*Structure analysis: 2026-05-24*
