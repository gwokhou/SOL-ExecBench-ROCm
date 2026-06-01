# Testing Patterns

**Analysis Date:** 2026-06-01

## Test Framework

**Runner:**
- Pytest `>=9.0.2`, declared in the `dev` dependency group in `pyproject.toml`.
- Parallel execution uses `pytest-xdist>=3.5`; `pyproject.toml` sets `addopts = "-n auto --dist loadgroup"`.
- Config: `pyproject.toml` for pytest options and core markers; `tests/conftest.py` for dynamic marker registration, hardware skip logic, and shared fixtures.

**Assertion Library:**
- Plain Python `assert` statements are the default assertion style.
- Use `pytest.raises` for exception paths, as in `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/test_matrix_claim_guardrails.py`, and `tests/sol_execbench/test_static_kernel_evidence.py`.
- Use `unittest.mock` only where call assertions or patch context managers are needed, as in `tests/sol_execbench/core/bench/test_clock_lock.py`.

**Run Commands:**
```bash
uv run pytest tests/                                      # Run all tests with configured xdist defaults
uv run pytest tests/sol_execbench/test_e2e.py             # Run one test file
uv run pytest tests -m timing_serial -n 0                 # Run serial GPU timing tests skipped by default
uv run pytest tests -m requires_rocm -n 0                 # Run live ROCm GPU tests on suitable hardware
uv run pytest tests/docker/dependencies/                  # Run ROCm container dependency checks
uv run ruff check .                                       # Lint Python source and tests
uv run ty check                                           # Type-check src and tests
```

## Test File Organization

**Location:**
- Package tests live under `tests/sol_execbench/`.
- Tests for modules with deep package ownership mirror the source path, for example `tests/sol_execbench/core/bench/test_eval_runtime.py`, `tests/sol_execbench/core/data/test_solution.py`, and `tests/sol_execbench/driver/test_problem_packager.py`.
- Broader project, migration, scoring, documentation, and CLI guardrails live as `tests/sol_execbench/test_*.py`.
- Example workflow coverage lives under `tests/examples/`, especially `tests/examples/test_examples.py` and `tests/examples/test_rocm_cli_paths.py`.
- Container dependency readiness checks live under `tests/docker/dependencies/`.
- Test-only malicious or sample kernels live under `tests/sol_execbench/samples/`.

**Naming:**
- Use `test_*.py` files and descriptive `test_*` functions.
- Group related behavior with `Test*` classes when it improves scanability, as in `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/core/data/test_solution.py`, and `tests/sol_execbench/core/bench/test_clock_lock.py`.
- Use stable parametrized IDs for dataset/example cases, such as the `test_id` fields in `tests/sol_execbench/test_e2e.py` and `tests/examples/test_examples.py`.

**Structure:**
```text
tests/
├── conftest.py                         # marker registration, ROCm skip logic, shared fixtures
├── sol_execbench_type_helpers.py        # typed Pydantic model constructors for tests
├── sol_execbench/
│   ├── core/bench/test_*.py             # bench helper unit tests
│   ├── core/data/test_*.py              # schema and validation tests
│   ├── driver/test_*.py                 # staging and template tests
│   ├── samples/                         # malicious/sample kernels and problem fixtures
│   └── test_*.py                        # CLI, scoring, docs, migration, and integration guardrails
├── examples/test_*.py                   # runnable examples and CLI-path tests
└── docker/dependencies/test_*.py        # ROCm container dependency checks
```

## Test Structure

**Suite Organization:**
```python
@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.test_id, marks=_mark_case(c)) for c in _CASES],
)
def test_e2e(tmp_path: Path, case: Sample):
    definition, solution, workloads = _load_sample(case.sample, case.solution_file)
    config = BenchmarkConfig(lock_clocks=False)

    pkg = ProblemPackager(
        definition=definition,
        workloads=workloads,
        solution=solution,
        config=config,
        output_dir=tmp_path / "staging",
        keep_output_dir=True,
    )

    cmd = pkg.execute()
    result = _run_subprocess(cmd, cwd=pkg.output_dir)
    assert result.returncode == 0, result.stderr
```

**Patterns:**
- Arrange test data as module constants, small dataclasses, or helper constructors. Examples: `_MINIMAL_DEFINITION` in `tests/sol_execbench/driver/test_eval_driver.py`, `Sample` in `tests/sol_execbench/test_e2e.py`, and `Example` in `tests/examples/test_examples.py`.
- Use `tmp_path` for all generated files, staging directories, sidecars, traces, and report output. Avoid writing into the repository tree from tests.
- Use `monkeypatch` for environment variables and injectable behavior. Examples include `tmp_cache_dir` in `tests/conftest.py`, environment snapshot tests in `tests/sol_execbench/test_cli_environment_snapshot.py`, and run-dataset CLI tests in `tests/examples/test_rocm_cli_paths.py`.
- Use `pytest.mark.parametrize` for schema matrices and status matrices, as in `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/test_dependency_matrix_classification.py`, and `tests/sol_execbench/test_static_kernel_evidence.py`.
- Use `pytest.param(..., marks=...)` when each case needs hardware or serial markers, as in `tests/sol_execbench/test_e2e.py` and `tests/examples/test_examples.py`.
- Use `pytest.mark.xdist_group("serial")` for tests that invoke GPU evaluation, subprocess templates, or other stateful paths that should not run concurrently.

## Mocking

**Framework:** Pytest `monkeypatch`, dependency injection via callable parameters, and `unittest.mock` for call-level assertions.

**Patterns:**
```python
def test_profile_collection_records_success_metadata(tmp_path):
    calls: list[list[str]] = []

    def runner(command: Sequence[str], cwd, timeout) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        (tmp_path / "profile.rocpd").write_text("profile db")
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout='{"definition": "demo"}\n',
            stderr="profiler note",
        )

    result = collect_rocprofv3_profile(request, runner=runner)
    assert result.succeeded is True
    assert calls[0][-3:] == ["--", "python", "eval_driver.py"]
```

```python
@pytest.fixture(autouse=True)
def _mock_rocm_smi_path(monkeypatch):
    monkeypatch.setattr(clock_lock_module.shutil, "which", lambda _tool: "rocm-smi")
```

**What to Mock:**
- Mock subprocess runners, executable discovery, clocks, environment variables, and current time at module boundaries. Follow `runner`, `which`, and `now` parameters in `src/sol_execbench/core/environment.py` and `src/sol_execbench/core/bench/rocm_profiler.py`.
- Mock ROCm hardware probes for CPU-safe behavior. `tests/conftest.py` exposes helper seams such as `_rocm_gpu_info(path_exists=...)`; tests under `tests/sol_execbench/test_rocm_marker_device_nodes.py` exercise these paths.
- Mock CLI sidecar collectors when asserting non-fatal metadata behavior, as in `tests/sol_execbench/test_cli_environment_snapshot.py`.
- Patch `subprocess.run` only where command construction and call ordering are the behavior under test, as in `tests/sol_execbench/core/bench/test_clock_lock.py`.

**What NOT to Mock:**
- Do not mock Pydantic model validation when schema behavior is the subject. Use real model constructors from `tests/sol_execbench_type_helpers.py`.
- Do not mock `ProblemPackager` in e2e and example workflow tests; those tests intentionally verify staging, generated templates, compile/execute command construction, and trace parsing.
- Do not fake marker semantics in normal tests. Use the registered markers and skip gates from `tests/conftest.py`.
- Do not mock GPU availability in tests marked `requires_rocm`; let collection skip when `/dev/kfd`, `/dev/dri`, PyTorch ROCm, or architecture requirements are unavailable.

## Fixtures and Factories

**Test Data:**
```python
def make_solution(**kwargs: Any) -> Solution:
    return Solution.model_validate(kwargs)


@pytest.fixture
def python_solution() -> Solution:
    return make_solution(**_PYTHON_SOLUTION_DICT)
```

**Location:**
- Shared pytest fixtures live in `tests/conftest.py`. The main reusable fixture is `tmp_cache_dir`, which sets `SOLEXECBENCH_CACHE_PATH` to a test-specific temporary directory.
- Typed schema constructors live in `tests/sol_execbench_type_helpers.py`.
- Large or repeated SOLAR derivation data lives in `tests/sol_execbench/solar_derivation_fixtures.py`.
- Problem and malicious-kernel samples live under `tests/sol_execbench/samples/`.
- Runnable public examples live under `examples/` and are exercised by `tests/examples/test_examples.py`.

## Coverage

**Requirements:** No coverage threshold is configured. `pyproject.toml` has no `[tool.coverage]` section, and `pytest-cov` is not declared in the `dev` dependency group.

**View Coverage:**
```bash
Not configured
```

## Test Types

**Unit Tests:**
- Schema validation and small deterministic helpers are tested without GPU access. Examples include `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/core/bench/test_utils.py`, `tests/sol_execbench/test_amd_bound_estimates.py`, and `tests/sol_execbench/test_dependency_matrix_classification.py`.
- Report and sidecar serialization tests write JSON/Markdown into `tmp_path` and assert stable fields, as in `tests/sol_execbench/test_consistency_script.py`, `tests/sol_execbench/test_claim_upgrade_script.py`, and `tests/sol_execbench/test_runtime_evidence_reports.py`.

**Integration Tests:**
- Driver subprocess tests stage `eval_driver.py`, `definition.json`, `workload.jsonl`, `solution.json`, and `kernel.py` into a temp directory, then run Python subprocesses. See `tests/sol_execbench/driver/test_eval_driver.py`.
- Packager tests verify staging files, cleanup behavior, HIP offload architecture injection, compile command construction, execution command construction, and trace parsing in `tests/sol_execbench/driver/test_problem_packager.py`.
- CLI and dataset integration tests invoke command entry points or subprocesses with generated temporary problem directories, as in `tests/examples/test_rocm_cli_paths.py` and `tests/sol_execbench/test_e2e.py`.

**E2E Tests:**
- Project e2e tests live in `tests/sol_execbench/test_e2e.py`. They load self-contained sample problems, package them, compile native HIP/C++ when required, execute, parse traces, and assert all workloads pass.
- Public example e2e tests live in `tests/examples/test_examples.py` and cover PyTorch, Triton, HIP/C++, hipBLAS, MIOpen, CK, rocWMMA, and compatibility-residue example paths.
- Live GPU e2e tests are marker-gated with `requires_rocm`, architecture markers, `cpp`, and `xdist_group("serial")`.

## Hardware and Marker Gates

**Configured Markers:**
- `cpp`: tests that compile HIP/C++ extensions.
- `requires_rocm`: tests requiring a ROCm GPU visible through PyTorch.
- `requires_rocm_dev`: tests requiring ROCm native extension development headers.
- `requires_ck`: tests requiring Composable Kernel headers.
- `requires_rocwmma`: tests requiring rocWMMA headers.
- `requires_rdna4`: tests requiring an AMD RDNA 4 GPU such as `gfx1200`.
- `requires_cdna3`: tests requiring an AMD CDNA 3 GPU such as `gfx942`.
- `requires_cutile`: legacy NVIDIA cuTile marker; always skipped in this ROCm-only port.
- `timing_serial`: GPU timing tests skipped unless explicitly selected with `-m timing_serial`.

**Skip Behavior:**
- `tests/conftest.py` checks `/dev/kfd` and `/dev/dri` before probing PyTorch ROCm on Linux.
- `requires_rocm` tests are skipped when PyTorch is not a ROCm build, no GPU is visible, or ROCm probing fails.
- `requires_rdna4` and `requires_cdna3` compare the detected gfx architecture with `gfx12*` or `gfx94*`.
- `requires_rocm_dev`, `requires_ck`, and `requires_rocwmma` check headers under `/opt/rocm`.
- `timing_serial` tests require explicit marker selection and should run with `-n 0`.

## Common Patterns

**Async Testing:**
```python
Not used. The project tests synchronous Python, subprocess, file, schema, and GPU execution paths.
```

**Error Testing:**
```python
with pytest.raises(ValidationError, match="HIP/C\\+\\+ and Python cannot be mixed"):
    _make_spec(languages=["pytorch", "hip_cpp"], entry_point="kernel.hip::run")
```

```python
result = subprocess.run(
    [sys.executable, "eval_driver.py"],
    cwd=tmp_path,
    capture_output=True,
    text=True,
    timeout=60,
)
assert result.returncode == 0, result.stderr
```

## CI Test Selection

**GitHub Actions:**
- `.github/workflows/code-quality.yml` runs on push and pull request for Python 3.12 and 3.13.
- CI installs with `uv sync --locked --all-groups`, then runs `uv run ruff check .`, `uv run ty check`, and CPU-safe pytest selections.
- CI runs `uv run pytest tests/sol_execbench --ignore=tests/sol_execbench/driver/test_eval_driver.py --ignore=tests/sol_execbench/test_e2e.py`.
- CI also runs `uv run pytest tests/examples/test_examples.py -k consistency`.
- CI does not run live ROCm GPU, Docker passthrough, or full e2e driver subprocess suites by default.

---

*Testing analysis: 2026-06-01*
