# Testing Patterns

**Analysis Date:** 2026-05-22

## Test Framework

**Runner:**
- Pytest `>=9.0.2` with `pytest-xdist>=3.5`, configured in `pyproject.toml`.
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`.
- Default addopts: `-n auto --dist loadgroup` in `pyproject.toml`.
- Collection hooks and shared fixtures: `tests/conftest.py`.

**Assertion Library:**
- Native `assert` statements with pytest introspection.
- `pytest.raises()` for expected exceptions, often with `match=...`.
- `unittest.mock` (`MagicMock`, `patch`, `call`) for subprocess/tooling mocks in files such as `tests/sol_execbench/core/bench/test_clock_lock.py`.

**Run Commands:**
```bash
uv run pytest tests/                                      # Run all tests with default xdist settings
uv run pytest tests/sol_execbench/test_e2e.py             # Run one test file
uv run pytest tests -m timing_serial -n 0                 # Run serial GPU timing tests
uv run pytest tests/examples/test_examples.py -m cpp      # Run example tests selected by marker
uv run ruff check .                                       # Run lint checks
```

## Test File Organization

**Location:**
- Package unit and integration tests live under `tests/sol_execbench/`.
- Tests for core data models mirror source paths: `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/core/data/test_definition.py`.
- Tests for benchmark helpers mirror source paths: `tests/sol_execbench/core/bench/test_io.py`, `tests/sol_execbench/core/bench/test_timing.py`, `tests/sol_execbench/core/bench/test_clock_lock.py`.
- Driver/template tests live under `tests/sol_execbench/driver/`: `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/driver/test_build_ext.py`, `tests/sol_execbench/driver/test_eval_driver.py`.
- Example workflow coverage lives under `tests/examples/test_examples.py`.
- Container dependency smoke tests live under `tests/docker/dependencies/`.
- Test sample kernels and JSON live under `tests/sol_execbench/samples/` and `tests/samples/`.

**Naming:**
- Test files use `test_*.py`.
- Test functions use descriptive `test_*` names that describe behavior: `test_legacy_cuda_nvidia_languages_rejected_with_guidance()` in `tests/sol_execbench/core/data/test_solution.py`.
- Class-based grouping uses `Test<Subject>` names without inheriting from `unittest.TestCase`: `TestLanguageValidation`, `TestCompileOptions`, `TestLockClocks`.
- Parametrized cases provide stable IDs via dataclass fields and `pytest.param(..., id=...)`, as in `tests/sol_execbench/test_e2e.py` and `tests/examples/test_examples.py`.

**Structure:**
```text
tests/
├── conftest.py                         # Hardware marker registration, skip policy, tmp cache fixture
├── docker/dependencies/                # Container/runtime dependency smoke tests
├── examples/test_examples.py           # Public example e2e tests
└── sol_execbench/
    ├── core/data/test_*.py             # Pydantic schema/model tests
    ├── core/bench/test_*.py            # Benchmark IO, timing, correctness, reward-hack tests
    ├── driver/test_*.py                # Packager and generated-template tests
    ├── test_e2e.py                     # Self-contained sample e2e tests
    └── test_*_audit.py                 # ROCm migration and public-contract guardrail tests
```

## Test Structure

**Suite Organization:**
```python
class TestLanguageValidation:
    """BuildSpec accepts only ROCm-native language categories."""

    @pytest.mark.parametrize("langs", [["pytorch"], ["triton"], ["pytorch", "triton"]])
    def test_pure_python_languages_accepted(self, langs):
        spec = _make_spec(languages=langs)
        assert set(spec.languages) == {SupportedLanguages(lg) for lg in langs}
```

**Patterns:**
- Define small local factory helpers near the tests they support: `_make_spec()` in `tests/sol_execbench/core/data/test_solution.py`, `_trace()` in `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py`.
- Use class groupings for related behaviors of one module or model: `TestSourceDiscovery`, `TestCompileOptions`, and `TestSoRename` in `tests/sol_execbench/driver/test_build_ext.py`.
- Use `@pytest.mark.parametrize` for schema matrices, dtype matrices, language/suffix compatibility, and example/sample catalogs.
- Use dataclasses for e2e case descriptors when cases carry several fields: `Sample`, `EvilCase`, and `Example` in `tests/sol_execbench/test_e2e.py` and `tests/examples/test_examples.py`.
- Assert rich failure messages for subprocess/e2e failures so stdout and stderr are visible: `tests/sol_execbench/test_e2e.py`, `tests/examples/test_examples.py`.
- Use AST parsing tests for generated templates and code contracts where runtime execution would be too broad: `tests/sol_execbench/driver/test_build_ext.py`.

## Mocking

**Framework:** `unittest.mock` plus pytest `monkeypatch`.

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
def test_env_var_overrides_preset(self, monkeypatch):
    monkeypatch.setenv("SOL_EXECBENCH_SCLK_LEVEL", "2")
    monkeypatch.setenv("SOL_EXECBENCH_MCLK_LEVEL", "3")
```

**What to Mock:**
- External commands and ROCm tooling: patch `subprocess.run` in `tests/sol_execbench/core/bench/test_clock_lock.py`.
- Sleeps and GPU clock verification to keep tests fast and deterministic: patch `time.sleep` and `verify_clocks` in `tests/sol_execbench/core/bench/test_clock_lock.py`.
- Import-time/runtime dependencies for generated templates: inject fake `torch.utils.cpp_extension` modules into `sys.modules` in `tests/sol_execbench/driver/test_build_ext.py`.
- Environment variables through `monkeypatch.setenv()` and `monkeypatch.delenv()`, as in `tests/conftest.py` and `tests/sol_execbench/core/bench/test_clock_lock.py`.
- Temporary build/cache directories via `tmp_path` and the shared `tmp_cache_dir` fixture in `tests/conftest.py`.

**What NOT to Mock:**
- Pydantic schema validation; instantiate actual models such as `BuildSpec`, `Solution`, `Definition`, and `Trace` in tests under `tests/sol_execbench/core/data/`.
- Public CLI help/options and JSON trace contracts; exercise real CLI or serialization behavior in `tests/sol_execbench/test_public_contract_guardrails.py` and `tests/sol_execbench/test_baseline_comparison.py`.
- End-to-end packaging flow when the test purpose is integration behavior; use `ProblemPackager` directly in `tests/sol_execbench/test_e2e.py` and `tests/examples/test_examples.py`.

## Fixtures and Factories

**Test Data:**
```python
def _make_spec(**overrides):
    base = dict(
        languages=["triton"],
        target_hardware=["LOCAL"],
        entry_point="kernel.py::run",
    )
    base.update(overrides)
    return BuildSpec(**base)
```

```python
@dataclass
class Example:
    test_id: str
    language: str
    problem: str
    solution_file: str
    expected_count: int
    extra_markers: list[str] = field(default_factory=list)
```

**Location:**
- Shared hardware skip logic and cache isolation live in `tests/conftest.py`.
- Module-specific factories live in the test module that uses them: `_make_solution_json()` in `tests/sol_execbench/driver/test_build_ext.py`, `_load_sample()` in `tests/sol_execbench/test_e2e.py`, `_load_example()` in `tests/examples/test_examples.py`.
- Sample problem directories live under `tests/sol_execbench/samples/` and are loaded by `tests/sol_execbench/test_e2e.py`.
- Public example directories live under `examples/` and are loaded by `tests/examples/test_examples.py`.

## Coverage

**Requirements:** No coverage threshold or `pytest-cov` configuration is detected in `pyproject.toml`.

**View Coverage:**
```bash
Not configured
```

## Test Types

**Unit Tests:**
- Schema/model tests cover validation matrices and public contract behavior: `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/core/data/test_definition.py`, `tests/sol_execbench/core/data/test_workload.py`, `tests/sol_execbench/core/data/test_dtypes.py`.
- Benchmark helper tests cover tensor generation, correctness metrics, timing helpers, clock locking, reward-hack detection, and IO behavior: `tests/sol_execbench/core/bench/test_io.py`, `tests/sol_execbench/core/bench/test_correctness.py`, `tests/sol_execbench/core/bench/test_timing.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`.
- Reporting/scoring tests cover derived summaries and guardrails: `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py`, `tests/sol_execbench/test_baseline_comparison.py`.

**Integration Tests:**
- Driver tests execute generated scripts in controlled temp directories: `tests/sol_execbench/driver/test_build_ext.py`, `tests/sol_execbench/driver/test_eval_driver.py`.
- Problem packaging tests cover staging, compile commands, execute commands, and stdout-to-trace parsing: `tests/sol_execbench/driver/test_problem_packager.py`.
- E2E tests package sample problems, compile HIP/C++ where needed, execute evaluation, parse traces, and assert successful workloads: `tests/sol_execbench/test_e2e.py`, `tests/examples/test_examples.py`.
- ROCm migration/public-contract audit tests inspect files and docs for contract drift and legacy CUDA/NVIDIA residue: `tests/sol_execbench/test_rocm_migration_residue_audit.py`, `tests/sol_execbench/test_rocm_test_suite_audit.py`, `tests/sol_execbench/test_public_contract_guardrails.py`.

**E2E Tests:**
- Framework: Pytest with subprocess execution.
- Main files: `tests/sol_execbench/test_e2e.py` and `tests/examples/test_examples.py`.
- Use `pytest.mark.xdist_group("serial")` for tests that should not run concurrently under xdist.
- Use `cpp` marker for tests that compile HIP/C++ extensions.

## Markers and Hardware Gates

**Configured Markers:**
- `cpp`: HIP/C++ extension compilation tests.
- `requires_rocm`: tests requiring a ROCm GPU visible through PyTorch.
- `requires_rdna4`: tests requiring AMD RDNA 4, such as `gfx1200`.
- `requires_cdna3`: tests requiring AMD CDNA 3, such as `gfx942`.
- `requires_cutile`: legacy NVIDIA cuTile marker that is skipped in this ROCm-only port.
- `timing_serial`: registered in `tests/conftest.py`; skipped by default unless `-m timing_serial` is explicitly selected.

**Skip Policy:**
- Hardware detection happens in `_rocm_gpu_info()` in `tests/conftest.py`.
- `pytest_collection_modifyitems()` in `tests/conftest.py` adds skip markers based on ROCm availability, detected architecture, legacy cuTile use, and timing selection.
- Timing tests in `tests/sol_execbench/core/bench/test_timing.py` set `pytestmark = pytest.mark.timing_serial`.

## Common Patterns

**Async Testing:**
```python
Not applicable; async test patterns are not used in this codebase.
```

**Error Testing:**
```python
with pytest.raises(ValidationError, match="HIP/C\\+\\+ and Python cannot be mixed"):
    _make_spec(languages=["pytorch", "hip_cpp"], entry_point="kernel.hip::run")
```

```python
with pytest.raises(RuntimeError, match="No HIP/C\\+\\+ source files"):
    _exec_build_ext(tmp_path)
```

**Subprocess Testing:**
```python
result = subprocess.run(
    cmd,
    cwd=cwd,
    capture_output=True,
    text=True,
    timeout=300,
)
assert result.returncode == 0, (
    f"Execution failed:\n  stdout={result.stdout}\n  stderr={result.stderr}"
)
```

**Serialization Testing:**
- Build dictionaries or JSONL files directly, then load through production models/utilities. Examples: `_write_jsonl()` in `tests/sol_execbench/test_baseline_comparison.py`, `_load_sample()` in `tests/sol_execbench/test_e2e.py`.
- Validate trace JSON stays strict and finite around NaN/Inf handling in `tests/sol_execbench/driver/test_eval_driver.py`.

---

*Testing analysis: 2026-05-22*
