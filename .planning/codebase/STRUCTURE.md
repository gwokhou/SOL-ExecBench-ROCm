# Codebase Structure

**Analysis Date:** 2026-05-22

## Directory Layout

```text
SOL-ExecBench-ROCm/
|-- src/
|   `-- sol_execbench/          # Python package source
|       |-- cli/                # Click command entry points
|       |-- core/               # Public models, benchmark helpers, reporting, diagnostics
|       |   |-- bench/          # Runtime input/correctness/timing/clock/reward-hack logic
|       |   |   `-- config/     # Benchmark and device config dataclasses
|       |   `-- data/           # Pydantic public schema models and JSON helpers
|       `-- driver/             # Staging/package layer and generated subprocess templates
|           `-- templates/      # build_ext.py and eval_driver.py copied into staging dirs
|-- tests/
|   |-- sol_execbench/          # Package tests, schema tests, driver tests, guardrails
|   |-- examples/               # Example workflow coverage
|   |-- samples/                # Test problem fixtures
|   `-- docker/                 # Docker dependency checks
|-- examples/                   # Runnable benchmark examples by implementation family
|-- docs/                       # Public schema, ROCm, parity, compliance docs
|-- docs/internal/              # Internal migration/readiness notes
|-- scripts/                    # Dataset/download/docker helper scripts
|-- docker/                     # ROCm evaluation container files
|-- data/                       # Downloaded benchmark assets; committed as placeholder only
|-- .planning/                  # GSD planning and generated codebase maps
|-- pyproject.toml              # Package, dependency, pytest, Ruff, uv configuration
|-- uv.lock                     # Locked dependency graph
|-- AGENTS.md                   # Repository and workflow instructions
`-- README.md                  # User-facing overview and quick start
```

## Directory Purposes

**`src/sol_execbench/`:**
- Purpose: Installable Python package.
- Contains: Package exports in `src/sol_execbench/__init__.py`, score helper in `src/sol_execbench/sol_score.py`, CLI, core, and driver subpackages.
- Key files: `src/sol_execbench/__init__.py`, `src/sol_execbench/sol_score.py`

**`src/sol_execbench/cli/`:**
- Purpose: User-facing command-line interfaces.
- Contains: Main benchmark CLI and baseline comparison CLI.
- Key files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/cli/baseline.py`, `src/sol_execbench/cli/__init__.py`

**`src/sol_execbench/core/`:**
- Purpose: Shared benchmark domain layer and public exports.
- Contains: Re-export surface, baseline comparison helpers, diagnostics, reporting, score guardrails, utility helpers, `bench/`, and `data/`.
- Key files: `src/sol_execbench/core/__init__.py`, `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/diagnostics.py`, `src/sol_execbench/core/reporting.py`, `src/sol_execbench/core/scoring_guardrails.py`, `src/sol_execbench/core/utils.py`

**`src/sol_execbench/core/data/`:**
- Purpose: Public schema definitions and serialization utilities.
- Contains: Pydantic models for definitions, workloads, solutions, traces, base-model helpers, dtype mapping, shape-expression resolver, and JSON/JSONL helpers.
- Key files: `src/sol_execbench/core/data/definition.py`, `src/sol_execbench/core/data/workload.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/data/json_utils.py`, `src/sol_execbench/core/data/dtypes.py`, `src/sol_execbench/core/data/shapes.py`

**`src/sol_execbench/core/bench/`:**
- Purpose: Runtime benchmark mechanics used by the generated evaluation driver.
- Contains: Clock-lock utilities, input generation, safetensors loading, output normalization, correctness metrics, timing, reward-hack defenses, and evaluation object helpers.
- Key files: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/core/bench/utils.py`, `src/sol_execbench/core/bench/clock_lock.py`

**`src/sol_execbench/core/bench/config/`:**
- Purpose: Dataclass configuration for benchmark behavior and known device clock presets.
- Contains: `BenchmarkConfig`, `ClockPreset`, preset lookup helpers.
- Key files: `src/sol_execbench/core/bench/config/benchmark_config.py`, `src/sol_execbench/core/bench/config/device_config.py`, `src/sol_execbench/core/bench/config/__init__.py`

**`src/sol_execbench/driver/`:**
- Purpose: Stage validated problems and submitted sources into executable subprocess directories.
- Contains: `ProblemPackager`, ROCm gfx/offload-arch helpers, package exports, and generated Python templates.
- Key files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/__init__.py`

**`src/sol_execbench/driver/templates/`:**
- Purpose: Scripts copied into the temporary staging directory and run as child processes.
- Contains: HIP/C++ build script and evaluation script.
- Key files: `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/driver/templates/eval_driver.py`

**`tests/sol_execbench/`:**
- Purpose: Package unit, integration, guardrail, migration, and driver template coverage.
- Contains: Top-level public contract/ROCm audit tests, nested `core/` tests, nested `driver/` tests, and fixture problem directories in `samples/`.
- Key files: `tests/sol_execbench/test_e2e.py`, `tests/sol_execbench/test_public_contract_guardrails.py`, `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/core/data/test_solution.py`

**`tests/examples/`:**
- Purpose: Validate example problem workflows.
- Contains: Example runner tests.
- Key files: `tests/examples/test_examples.py`

**`tests/samples/` and `tests/sol_execbench/samples/`:**
- Purpose: Local problem fixtures for package tests.
- Contains: Sample `definition.json`, `workload.jsonl`, `solution*.json`, source files, and reward-hack sample kernels.
- Key files: `tests/sol_execbench/samples/rmsnorm/definition.json`, `tests/sol_execbench/samples/evil_monkey_patch/kernel.py`, `tests/samples/rmsnorm/`

**`examples/`:**
- Purpose: Runnable benchmark examples grouped by implementation stack.
- Contains: Problem directories for PyTorch, Triton, HIP C++, ROCm/native-family examples, and legacy comparison material.
- Key files: `examples/hip_cpp/rmsnorm/solution_hip.json`, `examples/triton/rmsnorm/solution_triton.json`, `examples/pytorch/linear_backward/solution_python.json`

**`docs/`:**
- Purpose: Public documentation for schemas, ROCm support, original parity, compliance, solution format, trace format, and workload format.
- Contains: Markdown docs plus internal readiness and migration notes under `docs/internal/`.
- Key files: `docs/definition.md`, `docs/solution.md`, `docs/workload.md`, `docs/trace.md`, `docs/rocm.md`, `docs/original_parity.md`

**`scripts/`:**
- Purpose: Manual operational helpers.
- Contains: Dataset download scripts, batch dataset runner, and Docker launcher.
- Key files: `scripts/run_dataset.py`, `scripts/download_solexecbench.py`, `scripts/download_data.sh`, `scripts/run_docker.sh`

**`docker/`:**
- Purpose: ROCm/GPU container support.
- Contains: Dockerfile and container entrypoint.
- Key files: `docker/Dockerfile`, `docker/entrypoint.sh`

**`data/`:**
- Purpose: Local benchmark assets downloaded by scripts.
- Contains: Placeholder `.gitkeep`; real downloaded datasets should remain local.
- Key files: `data/.gitkeep`

**`.planning/`:**
- Purpose: GSD project planning, milestone, roadmap, state, and generated codebase maps.
- Contains: Project plans, milestone artifacts, research notes, and codebase documents.
- Key files: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`, `.planning/PROJECT.md`, `.planning/ROADMAP.md`

## Key File Locations

**Entry Points:**
- `src/sol_execbench/cli/main.py`: Main `sol-execbench` CLI.
- `src/sol_execbench/cli/baseline.py`: `sol-execbench-baseline` trace comparison CLI.
- `src/sol_execbench/driver/templates/build_ext.py`: Staged native ROCm build subprocess script.
- `src/sol_execbench/driver/templates/eval_driver.py`: Staged evaluation subprocess script.
- `scripts/run_dataset.py`: Batch dataset execution script.
- `scripts/run_docker.sh`: Docker wrapper script.

**Configuration:**
- `pyproject.toml`: Build backend, package metadata, dependencies, scripts, pytest markers, Ruff settings, uv indexes and sources.
- `uv.lock`: Resolved dependency lockfile.
- `.python-version`: Python version selector for local tooling.
- `.pre-commit-config.yaml`: Pre-commit hook configuration.
- `tests/conftest.py`: Pytest marker registration and ROCm hardware skip logic.
- `docker/Dockerfile`: ROCm evaluation image.
- `docker/entrypoint.sh`: Container startup/clock setup.

**Core Logic:**
- `src/sol_execbench/core/data/definition.py`: Definition schema, reference AST validation, axis/shape resolution helpers.
- `src/sol_execbench/core/data/solution.py`: Solution schema, supported ROCm languages, source validation, entrypoint validation, content hash.
- `src/sol_execbench/core/data/workload.py`: Workload schema, input unions, tolerance configuration.
- `src/sol_execbench/core/data/trace.py`: Trace/evaluation/result schema.
- `src/sol_execbench/driver/problem_packager.py`: Staging and compile/evaluate command generation.
- `src/sol_execbench/core/bench/io.py`: Input generation, safetensors loading, output normalization, DPS output allocation, timing allocator.
- `src/sol_execbench/core/bench/timing.py`: ROCm-compatible device-event timing.
- `src/sol_execbench/core/bench/correctness.py`: Numerical correctness and tolerance checks.
- `src/sol_execbench/core/bench/reward_hack.py`: Integrity and anti-cheat checks.
- `src/sol_execbench/core/baseline.py`: Baseline trace comparison.
- `src/sol_execbench/core/reporting.py`: Derived trace summaries.
- `src/sol_execbench/core/diagnostics.py`: ROCm environment/readiness helpers.

**Testing:**
- `tests/sol_execbench/core/data/`: Schema model tests.
- `tests/sol_execbench/core/bench/`: Benchmark helper tests.
- `tests/sol_execbench/driver/`: Problem packager and generated template tests.
- `tests/sol_execbench/test_e2e.py`: End-to-end benchmark workflow tests.
- `tests/sol_execbench/test_public_contract_guardrails.py`: Public CLI/schema guardrails.
- `tests/examples/test_examples.py`: Example workflow tests.

**Documentation:**
- `README.md`: Main project readme.
- `AGENTS.md`: Repository instructions for agents.
- `CONTRIBUTING.md`: Contribution, commit, and PR guidance.
- `SECURITY.md`: Security policy.
- `docs/definition.md`, `docs/solution.md`, `docs/workload.md`, `docs/trace.md`: Schema docs.
- `docs/rocm.md`, `docs/rocm_libraries.md`, `docs/original_parity.md`: ROCm and parity docs.

## Naming Conventions

**Files:**
- Python modules use `snake_case.py`: `problem_packager.py`, `scoring_guardrails.py`, `benchmark_config.py`.
- Tests use `test_*.py`: `tests/sol_execbench/core/bench/test_timing.py`.
- Example solution files use `solution_<language>.json`: `solution_python.json`, `solution_triton.json`, `solution_hip.json`.
- Benchmark problem files use canonical names: `definition.json`, `workload.jsonl`, optional `config.json`, and solution JSON files.
- Generated/staged template names are stable: `build_ext.py`, `eval_driver.py`, `benchmark_kernel.so`.

**Directories:**
- Python package directories are lowercase with underscores where needed: `sol_execbench`, `core`, `bench`, `data`, `driver`.
- Example directories are grouped by implementation family: `examples/pytorch/`, `examples/triton/`, `examples/hip_cpp/`.
- Problem fixture directories use descriptive operation names: `rmsnorm`, `linear_backward`, `gqa_paged_decode`.

## Where to Add New Code

**New CLI Option or Command:**
- Primary code: `src/sol_execbench/cli/main.py` for main benchmark behavior or `src/sol_execbench/cli/baseline.py` for baseline comparison behavior.
- Tests: `tests/sol_execbench/test_public_contract_guardrails.py` for public-contract changes plus focused tests under `tests/sol_execbench/`.
- Guidance: Keep the existing `sol-execbench` behavior stable; public CLI changes need docs and guardrail updates.

**New Public Schema Field or Model:**
- Primary code: `src/sol_execbench/core/data/`
- Re-export: `src/sol_execbench/core/data/__init__.py`, `src/sol_execbench/core/__init__.py`, and optionally `src/sol_execbench/__init__.py`.
- Docs: Matching file under `docs/` such as `docs/definition.md`, `docs/solution.md`, `docs/workload.md`, or `docs/trace.md`.
- Tests: `tests/sol_execbench/core/data/` plus public contract tests.

**New Native ROCm Solution Family:**
- Primary code: `src/sol_execbench/core/data/solution.py` for language enum/validation.
- Driver updates: `src/sol_execbench/driver/problem_packager.py` and `src/sol_execbench/driver/templates/eval_driver.py` native language sets.
- Build updates: `src/sol_execbench/driver/templates/build_ext.py` if compile behavior changes.
- Tests: `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/driver/test_build_ext.py`, `tests/sol_execbench/driver/test_eval_driver.py`.
- Examples: Add a problem under `examples/<family>/<operation>/`.

**New Benchmark Runtime Behavior:**
- Input/workload behavior: `src/sol_execbench/core/bench/io.py`.
- Correctness behavior: `src/sol_execbench/core/bench/correctness.py`.
- Timing behavior: `src/sol_execbench/core/bench/timing.py`.
- Reward-hack behavior: `src/sol_execbench/core/bench/reward_hack.py`.
- Evaluation orchestration: `src/sol_execbench/driver/templates/eval_driver.py`.
- Tests: Mirror the target module under `tests/sol_execbench/core/bench/` and add driver-template coverage when `eval_driver.py` behavior changes.

**New Reporting or Derived Evidence:**
- Primary code: `src/sol_execbench/core/reporting.py` or `src/sol_execbench/core/diagnostics.py`.
- Avoid: Adding non-canonical fields directly to `Trace` unless the public schema intentionally changes.
- Tests: Existing top-level tests under `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py` and diagnostics tests.

**New Baseline/Scoring Behavior:**
- Primary code: `src/sol_execbench/core/baseline.py`, `src/sol_execbench/core/scoring_guardrails.py`, or `src/sol_execbench/sol_score.py`.
- CLI code: `src/sol_execbench/cli/baseline.py`.
- Tests: `tests/sol_execbench/test_baseline_comparison.py`, `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py`.

**New Tests:**
- Schema and data model tests: `tests/sol_execbench/core/data/`.
- Bench helper tests: `tests/sol_execbench/core/bench/`.
- Driver/template tests: `tests/sol_execbench/driver/`.
- End-to-end behavior: `tests/sol_execbench/test_e2e.py`.
- Example workflow coverage: `tests/examples/test_examples.py`.
- ROCm hardware-specific tests: mark with `requires_rocm`, `requires_rdna4`, or `requires_cdna3` as registered in `tests/conftest.py`.

**New Example Problem:**
- Implementation files: `examples/<family>/<problem_name>/`.
- Required files: `definition.json`, `workload.jsonl`, `reference.py`, `kernel.py` or native sources, and `solution_<language>.json`.
- Tests: Extend `tests/examples/test_examples.py` when the example should be part of automated coverage.

**Utilities:**
- Shared benchmark utilities: `src/sol_execbench/core/bench/`.
- Shared schema/serialization utilities: `src/sol_execbench/core/data/`.
- Operational scripts: `scripts/`.
- Do not put package logic in `scripts/`; scripts should call package APIs or CLI commands.

## Special Directories

**`src/sol_execbench/driver/templates/`:**
- Purpose: Source templates copied into temporary staging directories.
- Generated: No; these are checked-in source templates.
- Committed: Yes.
- Rule: Keep them self-contained enough to run from an arbitrary staging directory. Use imports from installed/local `sol_execbench` and staged JSON/source files.

**Temporary staging directories:**
- Purpose: Runtime directories created by `tempfile.mkdtemp(prefix="sol_execbench_")` for each CLI run.
- Generated: Yes.
- Committed: No.
- Rule: These contain staged `definition.json`, `workload.jsonl`, `solution.json`, `config.json`, sources, templates, and optionally `benchmark_kernel.so`.

**`examples/`:**
- Purpose: Human-runnable example problems.
- Generated: No.
- Committed: Yes.
- Rule: Ruff excludes this directory; keep examples realistic and schema-valid.

**`tests/sol_execbench/samples/`:**
- Purpose: Test fixtures for package tests.
- Generated: No.
- Committed: Yes.
- Rule: Keep fixtures small and explicit; use reward-hack samples only for security/guardrail tests.

**`data/`:**
- Purpose: Download destination for benchmark assets.
- Generated: Yes for contents other than `.gitkeep`.
- Committed: Only `data/.gitkeep`.
- Rule: Do not commit downloaded datasets or generated benchmark outputs.

**`.planning/`:**
- Purpose: GSD project state and planning artifacts.
- Generated: Mixed.
- Committed: Project planning docs are tracked by workflow; code changes should not depend on runtime state here.

**`.venv/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`:**
- Purpose: Local environment and tool caches.
- Generated: Yes.
- Committed: No.
- Rule: Do not use these for architecture decisions or source edits.

---

*Structure analysis: 2026-05-22*
