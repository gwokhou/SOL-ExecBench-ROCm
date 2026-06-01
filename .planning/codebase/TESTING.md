# Testing Patterns

**Analysis Date:** 2026-06-01

## Test Framework

**Runner:**
- Pytest `>=9.0.2` configured in `pyproject.toml`.
- Parallel execution uses `pytest-xdist>=3.5`; default addopts are `-n auto --dist loadgroup` in `pyproject.toml`.
- Config: `pyproject.toml`

**Assertion Library:**
- Native `assert` statements and `pytest.raises` are the default assertion style.
- `unittest.mock` is used for subprocess and ROCm command mocking in tests such as `tests/sol_execbench/core/bench/test_clock_lock.py`.

**Run Commands:**
```bash
uv run pytest tests/                         # Run full test suite with default xdist settings
uv run pytest tests/sol_execbench/test_e2e.py # Run one test file
uv run pytest tests -m timing_serial -n 0     # Run serial GPU timing tests
uv run pytest -m requires_rocm -q -rs         # Run ROCm hardware-marked tests
uv run ruff check .                           # Lint
uv run ty check                               # Type check
```

## Test File Organization

**Location:**
- Package tests live under `tests/sol_execbench/`, mirroring source areas from `src/sol_execbench/`.
- Core bench tests live under `tests/sol_execbench/core/bench/`, such as `tests/sol_execbench/core/bench/test_eval_runtime.py` and `tests/sol_execbench/core/bench/test_timing.py`.
- Core data schema tests live under `tests/sol_execbench/core/data/`, such as `tests/sol_execbench/core/data/test_solution.py` and `tests/sol_execbench/core/data/test_definition.py`.
- Driver tests live under `tests/sol_execbench/driver/`, such as `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/driver/test_eval_driver.py`, and `tests/sol_execbench/driver/test_build_ext.py`.
- Example workflow tests live under `tests/examples/`, such as `tests/examples/test_examples.py` and `tests/examples/test_rocm_cli_paths.py`.
- Docker dependency smoke tests live under `tests/docker/dependencies/`, such as `tests/docker/dependencies/test_rocm_runtime.py`.

**Naming:**
- Use `test_*.py` files and `test_*` functions.
- Use class groupings for related behavior, such as `TestLanguageValidation`, `TestEntryPointSuffixValidation`, and `TestHardwareAndCompileOptions` in `tests/sol_execbench/core/data/test_solution.py`.
- Use dataclass descriptors and `pytest.param(..., id=...)` for table-driven e2e cases, as in `tests/sol_execbench/test_e2e.py` and `tests/examples/test_examples.py`.

**Structure:**
```text
tests/
├── conftest.py                         # Global hardware markers and shared fixtures
├── sol_execbench_type_helpers.py       # Typed model construction helpers
├── sol_execbench/
│   ├── core/bench/                     # Benchmark runtime, timing, correctness tests
│   ├── core/data/                      # Pydantic schema tests
│   ├── driver/                         # Staging, build, eval driver tests
│   ├── fixtures/                       # JSON fixtures for report/scoring tests
│   ├── samples/                        # Self-contained benchmark samples for e2e tests
│   └── test_*.py                       # Dataset, scoring, compatibility, and report tests
├── examples/                           # Tests that execute example problems
└── docker/dependencies/                # Container and ROCm dependency checks
```

## Test Structure

**Suite Organization:**
```python
@pytest.mark.parametrize(("legacy", "replacement"), LEGACY_LANGUAGE_REPLACEMENTS)
def test_legacy_cuda_nvidia_languages_rejected_with_guidance(self, legacy, replacement):
    with pytest.raises(ValidationError) as exc_info:
        _make_spec(languages=[legacy], entry_point="kernel.hip::run")

    message = str(exc_info.value)
    assert legacy in message
    assert replacement in message
```

**Patterns:**
- Build minimal inline dict fixtures for schemas and traces, then validate through helper constructors from `tests/sol_execbench_type_helpers.py`.
- Use `tmp_path` for all filesystem staging and generated artifacts, as in `tests/sol_execbench/driver/test_problem_packager.py` and `tests/sol_execbench/test_run_dataset_execution_closure.py`.
- Use `monkeypatch` for environment variables and function substitution, as in `tests/conftest.py`, `tests/sol_execbench/core/bench/test_make_eval_clock_warn.py`, and `tests/sol_execbench/test_run_dataset_execution_closure.py`.
- Use subprocess integration tests for generated drivers and scripts when behavior depends on process isolation, as in `tests/sol_execbench/driver/test_eval_driver.py` and `tests/sol_execbench/test_e2e.py`.
- Use explicit failure messages in assertions that involve subprocess output, traces, or hardware-sensitive behavior, as in `tests/sol_execbench/test_e2e.py` and `tests/docker/dependencies/test_rocm_runtime.py`.

## Mocking

**Framework:** `pytest.monkeypatch` and `unittest.mock`.

**Patterns:**
```python
with patch(f"{_MODULE}.subprocess.run", return_value=probe_result) as mock_run:
    result = probe_clock_lock_available()

assert result is True
mock_run.assert_called_once_with(
    ["sudo", "-n", "rocm-smi", "--showclocks"], capture_output=True
)
```

```python
def run_cli(*, workload_path: Path, **kwargs):
    calls.append(workload_path)
    return [_trace("selected-workload")]

monkeypatch.setattr(run_dataset, "run_cli", run_cli)
```

**What to Mock:**
- Mock ROCm command execution and `subprocess.run` calls in unit tests, as in `tests/sol_execbench/core/bench/test_clock_lock.py`.
- Mock environment variables with `monkeypatch.setenv` and `monkeypatch.delenv` for clock-lock and cache behavior, as in `tests/conftest.py` and `tests/sol_execbench/core/bench/test_make_eval_clock_warn.py`.
- Mock high-level script helper functions when testing control flow and provenance, as in `tests/sol_execbench/test_run_dataset_execution_closure.py`.
- Mock compile/load boundaries in build-template tests rather than compiling real native extensions, as in `tests/sol_execbench/driver/test_build_ext.py`.

**What NOT to Mock:**
- Do not mock Pydantic validation for schema tests; instantiate real models through helpers in `tests/sol_execbench_type_helpers.py`.
- Do not mock JSONL parsing and staging file creation in driver/package tests; use `tmp_path` and real files as in `tests/sol_execbench/driver/test_problem_packager.py`.
- Do not mock subprocess isolation in eval-driver integration tests; run `eval_driver.py` in a temporary directory as in `tests/sol_execbench/driver/test_eval_driver.py`.
- Do not force ROCm hardware availability in marker behavior tests; use marker skips from `tests/conftest.py` for real hardware checks.

## Fixtures and Factories

**Test Data:**
```python
def make_solution(**kwargs: Any) -> Solution:
    return Solution.model_validate(kwargs)
```

```python
@pytest.fixture
def python_solution() -> Solution:
    return make_solution(**_PYTHON_SOLUTION_DICT)
```

**Location:**
- Shared typed model factories live in `tests/sol_execbench_type_helpers.py`.
- Global pytest hooks and shared fixtures live in `tests/conftest.py`.
- JSON fixtures for SOL/Amd scoring and derivation live under `tests/sol_execbench/fixtures/`.
- Self-contained benchmark samples live under `tests/sol_execbench/samples/`.
- Example problem payloads live under `examples/` and are exercised from `tests/examples/test_examples.py`.

## Coverage

**Requirements:** No coverage threshold or coverage plugin is configured in `pyproject.toml`.

**View Coverage:**
```bash
uv run pytest tests/                         # Primary verification command
uv run pytest tests/sol_execbench -q          # Package tests only
uv run pytest tests/examples/test_examples.py # Example workflows
```

## Test Types

**Unit Tests:**
- Use unit tests for schema validation, helper functions, scoring calculations, timing policy, dependency classification, and report generation.
- Place unit tests next to source-layer mirrors, such as `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/core/bench/test_eval_runtime.py`, and `tests/sol_execbench/test_amd_native_score.py`.

**Integration Tests:**
- Use integration tests for staged evaluation, generated driver execution, CLI behavior, dataset execution closure, example execution, and Docker/ROCm dependency checks.
- Keep integration staging under `tmp_path` and assert stdout/stderr on failure, as in `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/test_e2e.py`, and `tests/examples/test_examples.py`.
- Use `pytest.mark.xdist_group("serial")` for tests that must not run concurrently, such as selected e2e paths in `tests/sol_execbench/test_e2e.py` and subprocess eval-driver tests in `tests/sol_execbench/driver/test_eval_driver.py`.

**E2E Tests:**
- E2E tests are pytest-based, not a separate browser or service framework.
- `tests/sol_execbench/test_e2e.py` packages sample definitions, workloads, and solutions through `ProblemPackager`, optionally compiles native HIP/C++ solutions, runs `eval_driver.py`, and asserts every trace passes.
- `tests/examples/test_examples.py` runs the public example problems through the same package/compile/execute pattern.
- Hardware-specific e2e tests must use markers such as `cpp`, `requires_rocm`, `requires_rocm_dev`, `requires_ck`, `requires_rocwmma`, `requires_rdna4`, or `requires_cdna3`.

## Common Patterns

**Async Testing:**
```python
# Async tests are not a current pattern in this repository.
# Prefer synchronous subprocess boundaries for CLI and driver workflows.
```

**Error Testing:**
```python
with pytest.raises(ValidationError, match="require a .py entry point"):
    _make_spec(languages=["pytorch"], entry_point="kernel.hip::run")
```

```python
result = subprocess.run(
    [sys.executable, "eval_driver.py"],
    cwd=tmp_path,
    env=env,
    capture_output=True,
    text=True,
    timeout=60,
)
assert result.returncode == 0, result.stderr
```

## Marker Policy

**Hardware and Toolchain Markers:**
- `cpp`: compiled HIP/C++ extension tests.
- `requires_rocm`: tests requiring ROCm GPU visibility through PyTorch.
- `requires_rocm_dev`: tests requiring ROCm HIP development headers.
- `requires_ck`: tests requiring Composable Kernel headers.
- `requires_rocwmma`: tests requiring rocWMMA headers.
- `requires_rdna4`: tests requiring AMD RDNA 4, such as `gfx1200`.
- `requires_cdna3`: tests requiring AMD CDNA 3, such as `gfx942`.
- `requires_cutile`: legacy NVIDIA cuTile marker; always skipped in this ROCm-only port.
- `timing_serial`: GPU timing tests skipped by default unless selected with `-m timing_serial`.

**Marker Implementation:**
- Marker registration and skip behavior live in `tests/conftest.py`.
- `pytest_collection_modifyitems` probes `/dev/kfd`, `/dev/dri`, PyTorch ROCm availability, `gcnArchName`, ROCm headers, CK headers, and rocWMMA headers before applying skips.
- Do not duplicate hardware probing in individual tests; add or reuse markers and let `tests/conftest.py` own skip reasons.

---

*Testing analysis: 2026-06-01*
