# Testing Patterns

**Analysis Date:** 2026-06-01

## Test Framework

**Runner:**
- Pytest `>=9.0.2`, configured in `[tool.pytest.ini_options]` in `pyproject.toml`.
- `pytest-xdist>=3.5` is enabled by default through `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`.
- Config: `pyproject.toml` and `tests/conftest.py`.

**Assertion Library:**
- Native `assert` statements and `pytest.raises`.
- Pydantic validation tests assert `pydantic.ValidationError` from models under `src/sol_execbench/core/data/`.
- Mocking uses `pytest.monkeypatch` and `unittest.mock` (`MagicMock`, `patch`, `call`) in tests such as `tests/sol_execbench/core/bench/test_clock_lock.py`.

**Run Commands:**
```bash
uv run pytest tests/                                      # Run all tests with default xdist settings
uv run pytest tests/sol_execbench/test_e2e.py             # Run one test file
uv run pytest -m timing_serial -n 0 tests/                # Run serialized timing tests that are skipped by default
uv run pytest -m requires_rocm -q -rs                     # Run ROCm GPU-marked tests and show skip reasons
uv run pytest tests/sol_execbench/core/data/test_solution.py -q  # Run a focused unit test module
```

## Test File Organization

**Location:**
- Core package tests live under `tests/sol_execbench/`, often mirroring `src/sol_execbench/`.
- Driver template tests live under `tests/sol_execbench/driver/`, such as `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/driver/test_build_ext.py`, and `tests/sol_execbench/driver/test_problem_packager.py`.
- Schema/model tests live under `tests/sol_execbench/core/data/`, such as `tests/sol_execbench/core/data/test_solution.py` and `tests/sol_execbench/core/data/test_definition.py`.
- Benchmark utility tests live under `tests/sol_execbench/core/bench/`, such as `tests/sol_execbench/core/bench/test_clock_lock.py`, `tests/sol_execbench/core/bench/test_timing.py`, and `tests/sol_execbench/core/bench/test_reward_hack.py`.
- Example workflow tests live under `tests/examples/`, such as `tests/examples/test_examples.py` and `tests/examples/test_rocm_cli_paths.py`.
- Docker dependency and runtime checks live under `tests/docker/dependencies/`.

**Naming:**
- Test modules use `test_*.py`.
- Test functions use descriptive behavior names, such as `test_dangerous_native_compile_options_rejected` in `tests/sol_execbench/core/data/test_solution.py`.
- Test classes group behavior by component or feature with `Test*` names, such as `TestLockClocks` in `tests/sol_execbench/core/bench/test_clock_lock.py`.

**Structure:**
```text
tests/
|-- conftest.py                         # Marker registration, hardware skips, shared tmp cache fixture
|-- sol_execbench_type_helpers.py       # Typed model construction helpers for tests
|-- sol_execbench/
|   |-- core/data/test_*.py             # Pydantic schema and contract tests
|   |-- core/bench/test_*.py            # Benchmark utility, timing, correctness, reward-hack tests
|   |-- driver/test_*.py                # Packaging and generated driver template tests
|   `-- test_*.py                       # CLI, dataset, scoring, docs, matrix, and release guardrail tests
|-- examples/test_*.py                  # Runnable example workflow coverage
`-- docker/dependencies/test_*.py       # Docker and ROCm dependency checks
```

## Test Structure

**Suite Organization:**
```python
class TestLanguageValidation:
    @pytest.mark.parametrize("langs", [["pytorch"], ["triton"], ["pytorch", "triton"]])
    def test_pure_python_languages_accepted(self, langs):
        spec = _make_spec(languages=langs)
        assert set(spec.languages) == {SupportedLanguages(lg) for lg in langs}

    def test_legacy_cuda_nvidia_languages_rejected_with_guidance(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_spec(languages=["cuda_cpp"], entry_point="kernel.hip::run")

        message = str(exc_info.value)
        assert "cuda_cpp" in message
        assert "hip_cpp" in message
```

**Patterns:**
- Build compact local fixtures with private helper functions such as `_make_spec()` in `tests/sol_execbench/core/data/test_solution.py`, `_definition()` in `tests/sol_execbench/test_dataset_runner.py`, and `_run_eval_driver()` in `tests/sol_execbench/driver/test_eval_driver.py`.
- Use `pytest.mark.parametrize` for matrix-style schema and behavior checks, especially language, hardware target, dtype, and report validation tests.
- Use `pytest.param(..., id=..., marks=...)` when dynamically assigning hardware markers, as in `tests/examples/test_examples.py`.
- Keep subprocess integration tests explicit about staging files and environment, as in `_run_eval_driver_process()` in `tests/sol_execbench/driver/test_eval_driver.py`.
- For guardrail tests that inspect project files, read exact paths through `Path` and assert required/forbidden phrases, as in `tests/sol_execbench/test_rocm_test_suite_audit.py`.

## Mocking

**Framework:** `pytest.monkeypatch` and `unittest.mock`.

**Patterns:**
```python
def test_run_cli_parses_jsonl_and_ignores_non_json_stdout(tmp_path, monkeypatch):
    trace = {"evaluation": {"status": "PASSED"}}

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="noise\n" + json.dumps(trace) + "\n",
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    traces = runner.run_cli(
        definition_path=Path("definition.json"),
        workload_path=Path("workload.jsonl"),
        solution_path=Path("solution.json"),
        output_dir=tmp_path,
        job_name="demo",
        timeout=1,
    )

    assert traces == [trace]
```

**What to Mock:**
- Mock subprocess boundaries, filesystem probes, sleeps, and external tool paths. Examples: patch `subprocess.run`, `time.sleep`, and `shutil.which` in `tests/sol_execbench/core/bench/test_clock_lock.py`.
- Use `monkeypatch.setenv()` and `monkeypatch.delenv()` for environment-sensitive behavior, as in `tests/conftest.py`, `tests/sol_execbench/core/bench/test_make_eval_clock_warn.py`, and `tests/sol_execbench/core/bench/test_clock_lock.py`.
- Mock CLI runners or collector functions for dataset orchestration tests, as in `tests/sol_execbench/test_dataset_runner.py` and `tests/examples/test_rocm_cli_paths.py`.

**What NOT to Mock:**
- Do not mock Pydantic validation in schema tests; construct models through `make_*` helpers from `tests/sol_execbench_type_helpers.py` and assert real validation behavior.
- Do not mock generated driver syntax. Validate generated templates with `ast.parse()` as in `tests/sol_execbench/driver/test_eval_driver.py` and `tests/sol_execbench/driver/test_build_ext.py`.
- Do not bypass hardware marker gates. Mark ROCm, native extension, CK, rocWMMA, RDNA4, CDNA3, and timing tests so `tests/conftest.py` can skip them consistently.

## Fixtures and Factories

**Test Data:**
```python
def make_solution(**kwargs: Any) -> Solution:
    return Solution.model_validate(kwargs)

def make_build_spec(**kwargs: Any) -> BuildSpec:
    return BuildSpec.model_validate(kwargs)
```

**Location:**
- Shared model factories live in `tests/sol_execbench_type_helpers.py`.
- Shared pytest configuration and fixtures live in `tests/conftest.py`.
- `tmp_cache_dir` in `tests/conftest.py` sets `SOLEXECBENCH_CACHE_PATH` to an isolated temporary directory for tests that build or cache artifacts.
- Larger sample kernels and malicious-solution fixtures live under `tests/sol_execbench/samples/`.
- Example problem fixtures live under `examples/` and are exercised through `tests/examples/test_examples.py`.

## Coverage

**Requirements:** No coverage threshold or coverage tool configuration is detected in `pyproject.toml`, `.coveragerc`, or repository-level coverage config files.

**View Coverage:**
```bash
uv run pytest tests/                                      # Primary verification path
uv run pytest tests/sol_execbench/core/data/test_solution.py -q  # Focused schema regression path
```

## Test Types

**Unit Tests:**
- Schema and validator tests under `tests/sol_execbench/core/data/` check Pydantic model behavior, error messages, dtype conversion, workload helpers, and ROCm schema constraints.
- Utility tests under `tests/sol_execbench/core/bench/` check correctness helpers, timing policy, IO generation, clock locking, reward-hack detection, and profiler parsing.
- Scoring and reporting tests under `tests/sol_execbench/test_*.py` check deterministic JSON/Markdown reports, guardrails, matrix semantics, AMD scoring, and contract behavior.

**Integration Tests:**
- Driver subprocess tests in `tests/sol_execbench/driver/test_eval_driver.py` create staging files, execute generated `eval_driver.py`, and parse JSONL traces without requiring the full CLI.
- Problem packaging and build template tests in `tests/sol_execbench/driver/test_problem_packager.py` and `tests/sol_execbench/driver/test_build_ext.py` validate generated commands and native build behavior.
- Dataset and CLI workflow tests in `tests/sol_execbench/test_dataset_runner.py`, `tests/sol_execbench/test_run_dataset_execution_closure.py`, and `tests/examples/test_rocm_cli_paths.py` validate orchestration and persistence behavior with temp directories.

**E2E Tests:**
- Example E2E tests live in `tests/examples/test_examples.py` and package example problems from `examples/`.
- ROCm CLI path E2E tests live in `tests/examples/test_rocm_cli_paths.py` and are marked `requires_rocm` plus `xdist_group("serial")`.
- Project-level E2E coverage also exists in `tests/sol_execbench/test_e2e.py`.

## Hardware and Marker Policy

**Markers:**
- `cpp`: tests that compile HIP/C++ extensions.
- `requires_rocm`: tests that require a ROCm GPU visible through PyTorch.
- `requires_rocm_dev`: tests that require ROCm HIP development headers, registered in `tests/conftest.py`.
- `requires_ck`: tests that require Composable Kernel headers, registered in `tests/conftest.py`.
- `requires_rocwmma`: tests that require rocWMMA headers, registered in `tests/conftest.py`.
- `requires_rdna4`: tests that require AMD RDNA 4, such as gfx1200.
- `requires_cdna3`: tests that require AMD CDNA 3, such as gfx942.
- `requires_cutile`: legacy NVIDIA cuTile marker that is skipped in this ROCm-only port.
- `timing_serial`: GPU timing tests skipped by default unless selected with `-m timing_serial`.

**Pattern:**
- Add hardware-sensitive tests with the narrowest marker set needed.
- Use `pytest.mark.xdist_group("serial")` for tests that must not race on GPU resources or generated staging behavior.
- Let `tests/conftest.py` decide skip reasons by probing `/dev/kfd`, `/dev/dri`, PyTorch ROCm availability, ROCm development headers, CK headers, and rocWMMA headers.

## Common Patterns

**Async Testing:**
```python
# No asyncio-specific test pattern is detected.
# Subprocess and hardware workflows are tested synchronously with explicit timeouts.
result = subprocess.run(
    [sys.executable, "eval_driver.py"],
    cwd=tmp_path,
    env=env,
    capture_output=True,
    text=True,
    timeout=60,
)
```

**Error Testing:**
```python
with pytest.raises(ValidationError, match="require a .py entry point"):
    _make_spec(languages=["triton"], entry_point="kernel.hip::run")
```

---

*Testing analysis: 2026-06-01*
