# Testing Patterns

**Analysis Date:** 2026-05-31

## Test Framework

**Runner:**
- Pytest `>=9.0.2`
- Config: `pyproject.toml`
- Parallelism: `pytest-xdist>=3.5` with default `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`
- Hardware-aware collection and shared fixtures: `tests/conftest.py`

**Assertion Library:**
- Plain Python `assert`
- `pytest.raises` for error paths
- `unittest.mock` for subprocess and dependency boundary assertions

**Run Commands:**
```bash
uv run pytest tests/              # Run all tests with default xdist settings
uv run pytest tests/sol_execbench/test_e2e.py  # Run one focused test file
uv run pytest tests -m timing_serial -n 0      # Run serial GPU timing tests
uv run pytest tests -m requires_rocm -n 0      # Run ROCm GPU-marked tests
uv run ty check                   # Type-check src and tests
uv run ruff check .               # Lint source and tests
```

## Test File Organization

**Location:**
- Package tests live under `tests/sol_execbench/`.
- Source-near package subareas mirror core modules, such as `tests/sol_execbench/core/data/`, `tests/sol_execbench/core/bench/`, and `tests/sol_execbench/driver/`.
- Example workflow tests live under `tests/examples/`.
- Container and ROCm dependency readiness checks live under `tests/docker/dependencies/`.
- Shared helpers live in `tests/conftest.py`, `tests/sol_execbench_type_helpers.py`, and `tests/sol_execbench/solar_derivation_fixtures.py`.
- Sample benchmark fixtures live under `tests/sol_execbench/samples/`.

**Naming:**
- Use `test_*.py` files and descriptive `test_*` functions, such as `test_legacy_cuda_nvidia_languages_rejected_with_guidance` in `tests/sol_execbench/core/data/test_solution.py`.
- Use `Test*` classes to group related behavior when a file covers a broad API, such as `TestLanguageValidation` in `tests/sol_execbench/core/data/test_solution.py` and `TestLockClocks` in `tests/sol_execbench/core/bench/test_clock_lock.py`.
- Use helper names prefixed with `_` for local factories and subprocess runners, such as `_make_spec` in `tests/sol_execbench/core/data/test_solution.py` and `_run_eval_driver` in `tests/sol_execbench/driver/test_eval_driver.py`.

**Structure:**
```
tests/
+-- conftest.py                         # hardware marker registration, skip logic, shared tmp_cache_dir fixture
+-- sol_execbench_type_helpers.py       # typed Pydantic model constructors for tests
+-- sol_execbench/
|   +-- core/data/test_*.py             # schema and model validation tests
|   +-- core/bench/test_*.py            # timing, correctness, IO, clock, reward-hack tests
|   +-- driver/test_*.py                # generated driver/build subprocess coverage
|   +-- samples/                        # benchmark sample fixtures
|   +-- test_*.py                       # CLI, scoring, matrix, docs, evidence, E2E guardrails
+-- examples/test_*.py                  # example consistency and workflow tests
+-- docker/dependencies/test_*.py       # ROCm container dependency checks
```

## Test Structure

**Suite Organization:**
```python
class TestLanguageValidation:
    @pytest.mark.parametrize("langs", [["pytorch"], ["triton"], ["pytorch", "triton"]])
    def test_pure_python_languages_accepted(self, langs):
        spec = _make_spec(languages=langs)
        assert set(spec.languages) == {SupportedLanguages(lg) for lg in langs}

    def test_cuda_cflags_rejected_with_hip_cflags_guidance(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_spec(
                languages=["hip_cpp"],
                entry_point="kernel.hip::run",
                compile_options={"cuda_cflags": ["-O3"]},
            )
        assert "cuda_cflags" in str(exc_info.value)
        assert "hip_cflags" in str(exc_info.value)
```

**Patterns:**
- Use Arrange/Act/Assert spacing rather than explicit comments for simple tests, as in `tests/sol_execbench/test_cli_environment_snapshot.py`.
- Use `pytest.mark.parametrize` for schema matrix checks, dtype mappings, language categories, Docker target classification, and report variants.
- Use `tmp_path` for isolated file output, staged benchmark problems, sidecars, and generated reports.
- Use `monkeypatch` for environment variables, module attribute replacement, CLI `sys.argv`, and injected failures.
- Use module-level sample dictionaries for repeated subprocess fixtures, as in `_MINIMAL_DEFINITION`, `_MINIMAL_WORKLOAD`, and `_SOLUTION_SPEC` in `tests/sol_execbench/driver/test_eval_driver.py`.
- Use marker gates for hardware-sensitive tests. `tests/conftest.py` registers and applies skips for `requires_rocm`, `requires_rocm_dev`, `requires_ck`, `requires_rocwmma`, `requires_rdna4`, `requires_cdna3`, `requires_cutile`, and `timing_serial`.

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
def collector() -> EnvironmentSnapshot:
    raise RuntimeError("probe failed")

written = cli_main._write_environment_snapshot_sidecar(
    tmp_path / "trace.jsonl",
    collector=collector,
)

assert written is None
```

**What to Mock:**
- Mock subprocess calls and ROCm tooling boundaries in unit tests, as in `tests/sol_execbench/core/bench/test_clock_lock.py`.
- Mock filesystem roots, cache directories, environment variables, and CLI arguments with `tmp_path` and `monkeypatch`.
- Mock optional collectors or injected runner functions instead of invoking real host probes, as in `tests/sol_execbench/test_cli_environment_snapshot.py` and `src/sol_execbench/core/environment.py`.
- Mock expensive or unavailable GPU behavior for CPU-safe unit tests; reserve live execution for marker-gated tests.

**What NOT to Mock:**
- Do not mock Pydantic validation when testing schema contracts; instantiate real models through constructors or helpers in `tests/sol_execbench_type_helpers.py`.
- Do not mock generated driver syntax checks; parse or execute the actual template from `src/sol_execbench/driver/templates/eval_driver.py` through helpers in `tests/sol_execbench/driver/test_eval_driver.py`.
- Do not mock public docs/guardrail files when tests assert wording, schemas, or examples; read the committed files directly.
- Do not run unmarked live ROCm, Docker, or GPU timing work in CPU-safe tests.

## Fixtures and Factories

**Test Data:**
```python
def make_build_spec(**kwargs: Any) -> BuildSpec:
    return BuildSpec.model_validate(kwargs)

def _make_spec(**overrides):
    base = dict(
        languages=["triton"],
        target_hardware=["LOCAL"],
        entry_point="kernel.py::run",
    )
    base.update(overrides)
    return make_build_spec(**base)
```

**Location:**
- Use `tests/sol_execbench_type_helpers.py` for typed constructors around Pydantic `model_validate`.
- Use `tests/conftest.py` for shared hardware marker behavior and the `tmp_cache_dir` fixture that sets `SOLEXECBENCH_CACHE_PATH`.
- Use `tests/sol_execbench/samples/` for reusable benchmark problem fixtures.
- Use test-local constants when the data only supports one suite, such as `_MINIMAL_DEFINITION` in `tests/sol_execbench/driver/test_eval_driver.py`.

## Coverage

**Requirements:** No coverage threshold is configured in `pyproject.toml`, and no `.coveragerc` or coverage-specific config is present.

**View Coverage:**
```bash
uv run --with coverage coverage run --source=src/sol_execbench -m pytest tests/
uv run --with coverage coverage report
```

## Test Types

**Unit Tests:**
- Schema and model tests under `tests/sol_execbench/core/data/` instantiate real Pydantic models and assert accepted/rejected payloads.
- Core helper tests under `tests/sol_execbench/core/bench/` validate timing summaries, IO, correctness, reward-hack detection, and clock-lock command behavior.
- Scoring, matrix, dataset, and report tests under `tests/sol_execbench/test_*.py` validate deterministic JSON shapes, claim boundaries, and guardrails.

**Integration Tests:**
- Driver integration tests under `tests/sol_execbench/driver/` stage temporary files and run generated templates or subprocesses.
- CLI/evidence tests under `tests/sol_execbench/test_cli_environment_snapshot.py` and related files exercise sidecar writing and public command behavior.
- Example tests under `tests/examples/` validate example consistency and runnable workflows.
- Docker dependency tests under `tests/docker/dependencies/` verify ROCm runtime, HIP, PyTorch ROCm, Triton ROCm, and library availability inside a suitable container.

**E2E Tests:**
- `tests/sol_execbench/test_e2e.py` and selected `tests/examples/` tests cover end-to-end benchmark execution.
- `tests/sol_execbench/test_e2e.py` and `tests/sol_execbench/driver/test_eval_driver.py` are excluded from CI CPU-safe package runs in `.github/workflows/code-quality.yml`.
- Live GPU E2E or timing tests must be marker-filtered and commonly run with `-n 0` to avoid parallel GPU interference.

## Common Patterns

**Async Testing:**
```python
result = subprocess.run(
    [sys.executable, "eval_driver.py"],
    cwd=tmp_path,
    env=env,
    capture_output=True,
    text=True,
    timeout=60,
)
traces = parse_eval_result(result.stdout, result.stderr)
assert traces[0]["evaluation"]["status"] == "PASSED"
```

**Error Testing:**
```python
with pytest.raises(ValidationError, match="require a .py entry point"):
    _make_spec(languages=["triton"], entry_point="kernel.hip::run")
```

```python
with patch(f"{_MODULE}.subprocess.run", side_effect=FileNotFoundError):
    assert probe_clock_lock_available() is False
```

---

*Testing analysis: 2026-05-31*
