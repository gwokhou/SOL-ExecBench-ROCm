# Testing Patterns

**Analysis Date:** 2026-05-24

## Test Framework

**Runner:**
- Pytest `>=9.0.2` from the `dev` dependency group in `pyproject.toml`.
- Parallel execution uses `pytest-xdist >=3.5` with `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`.
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`.

**Assertion Library:**
- Plain Python `assert` statements are the standard assertion style.
- Use `pytest.raises` for exception paths, as in `tests/sol_execbench/core/data/test_solution.py` and `tests/sol_execbench/driver/test_build_ext.py`.
- Use `click.testing.CliRunner` for Click CLI checks, as in `tests/sol_execbench/test_contract.py`.
- Use `unittest.mock.MagicMock` only when a dependency must be isolated, as in `tests/sol_execbench/driver/test_build_ext.py`.

**Run Commands:**
```bash
uv run pytest tests/                                      # Run all tests with configured xdist
uv run pytest tests/sol_execbench/test_e2e.py             # Run one test file
uv run pytest tests -m timing_serial -n 0                 # Run GPU timing tests that are skipped by default
uv run pytest tests/sol_execbench/core/data/test_solution.py -q  # Run a focused unit test file
```

## Test File Organization

**Location:**
- Main package tests live in `tests/sol_execbench/`.
- Tests for nested package areas may mirror the source tree, such as `tests/sol_execbench/core/data/test_solution.py` for `src/sol_execbench/core/data/solution.py` and `tests/sol_execbench/core/bench/test_io.py` for `src/sol_execbench/core/bench/io.py`.
- Driver/template integration tests live in `tests/sol_execbench/driver/`.
- Runnable example coverage lives in `tests/examples/test_examples.py`.
- Docker dependency checks live in `tests/docker/dependencies/`.
- E2E sample assets live under `tests/sol_execbench/samples/`.

**Naming:**
- Use `test_*.py` files and `test_*` functions.
- Use descriptive names that encode behavior and expected result: `test_static_source_review_blocks_precision_downgrade` in `tests/sol_execbench/driver/test_eval_driver.py`, `test_evaluator_contract_versions_are_stable` in `tests/sol_execbench/test_contract.py`.
- Use `Test*` classes to group related schema or template behavior without per-class mutable setup, as in `tests/sol_execbench/core/data/test_solution.py` and `tests/sol_execbench/driver/test_build_ext.py`.

**Structure:**
```text
tests/
├── conftest.py                         # ROCm marker registration, skip policy, shared fixtures
├── docker/dependencies/                # Container/runtime dependency checks
├── examples/test_examples.py           # Example workflow coverage
└── sol_execbench/
    ├── core/data/test_*.py             # Data schema tests
    ├── core/bench/test_*.py            # Benchmark helper tests
    ├── driver/test_*.py                # Template and subprocess driver tests
    ├── samples/                        # E2E fixture problem directories
    └── test_*.py                       # Contract, scoring, ROCm audit, and E2E tests
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
- Keep reusable fixtures and builders private to the test module: `_make_spec()` in `tests/sol_execbench/core/data/test_solution.py`, `_run_eval_driver()` in `tests/sol_execbench/driver/test_eval_driver.py`, `_load_sample()` in `tests/sol_execbench/test_e2e.py`.
- Use dataclasses for parametrized case descriptors when a test matrix has multiple fields: `Sample` and `EvilCase` in `tests/sol_execbench/test_e2e.py`.
- Use `pytest.param(..., id=..., marks=...)` for readable parametrized IDs and per-case markers, as in `tests/sol_execbench/test_e2e.py`.
- Include high-signal assertion messages for subprocess and E2E failures, especially stdout/stderr and trace logs, as in `tests/sol_execbench/test_e2e.py`.
- Test public contracts by asserting exact schema versions, field lists, and enum values, as in `tests/sol_execbench/test_contract.py`.
- Test generated templates by parsing them with `ast.parse` before executing behavior checks, as in `tests/sol_execbench/driver/test_eval_driver.py` and `tests/sol_execbench/driver/test_build_ext.py`.

## Mocking

**Framework:** `pytest` monkeypatch fixtures, dependency injection, `unittest.mock.MagicMock`, local fake functions.

**Patterns:**
```python
def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    calls.append(list(command))
    (tmp_path / "timing.csv").write_text(ROCPROFV3_CSV)
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=0,
        stdout="profiled",
        stderr="",
    )
```

**What to Mock:**
- Mock external command runners by passing a fake `runner` callable, as in `tests/sol_execbench/test_rocm_profiler.py`.
- Use `monkeypatch.setattr` for script-level functions such as `run_dataset.run_cli`, as in `tests/sol_execbench/test_run_dataset_amd_score.py` and `tests/sol_execbench/test_run_dataset_execution_closure.py`.
- Use `monkeypatch.setenv` for environment-dependent behavior, as in `tests/conftest.py` and `tests/sol_execbench/core/bench/test_make_eval_clock_warn.py`.
- Use `MagicMock` and temporary `sys.modules` stubs when executing build templates without a real PyTorch extension toolchain, as in `tests/sol_execbench/driver/test_build_ext.py`.
- Use fake filesystem inputs under `tmp_path` for manifests, JSONL, safetensors references, profiler CSV, and staging directories.

**What NOT to Mock:**
- Do not mock Pydantic model validation in schema tests; instantiate real models such as `BuildSpec`, `Definition`, `Workload`, `Solution`, and `Trace`.
- Do not mock subprocess boundaries in driver and E2E tests that are explicitly verifying process packaging and execution; `tests/sol_execbench/driver/test_eval_driver.py` runs `eval_driver.py` with `subprocess.run`.
- Do not force ROCm hardware availability. Use markers and skip logic from `tests/conftest.py`.

## Fixtures and Factories

**Test Data:**
```python
_MINIMAL_DEFINITION = {
    "name": "test_vecadd",
    "op_type": "elementwise",
    "axes": {"n": {"type": "const", "value": 64}},
    "inputs": {
        "x": {"shape": ["n"], "dtype": "float32"},
        "y": {"shape": ["n"], "dtype": "float32"},
    },
    "outputs": {"z": {"shape": ["n"], "dtype": "float32"}},
    "reference": "import torch\ndef run(x, y):\n    return x + y",
}
```

**Location:**
- Shared pytest fixtures live in `tests/conftest.py`; `tmp_cache_dir` creates an isolated `SOLEXECBENCH_CACHE_PATH`.
- Module-local dictionaries are used for compact schema fixtures: `_MINIMAL_DEFINITION` in `tests/sol_execbench/driver/test_eval_driver.py`, `_EVIL_DEFINITION_DICT` in `tests/sol_execbench/test_e2e.py`.
- Larger reusable fixtures live in helper modules such as `tests/sol_execbench/solar_derivation_fixtures.py`.
- File-based sample problems live under `tests/sol_execbench/samples/`.
- JSON sample solutions for reward-hack checks live under `tests/samples/rmsnorm/`.

## Coverage

**Requirements:** No numeric coverage target or coverage config file is detected.

**View Coverage:**
```bash
uv run pytest tests/                # Primary verification command
```

## Test Types

**Unit Tests:**
- Scope: pure schema validation, enum contracts, scoring formulas, parser behavior, shape/dtype utilities, and guardrails.
- Examples: `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/test_contract.py`, `tests/sol_execbench/test_rocm_profiler.py`, `tests/sol_execbench/test_amd_native_score.py`.
- Approach: instantiate real models and assert exact values, warnings, statuses, and exception messages.

**Integration Tests:**
- Scope: subprocess evaluation, problem packaging, template execution, CLI output, profiler command construction, dataset scripts, and generated trace parsing.
- Examples: `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/test_e2e.py`, `tests/sol_execbench/test_run_dataset_execution_closure.py`.
- Approach: write temporary staging files with `tmp_path`, run real commands or injected runners, then parse JSON/JSONL outputs into model objects.

**E2E Tests:**
- Framework: pytest plus subprocess execution.
- Main E2E file: `tests/sol_execbench/test_e2e.py`.
- Sample-based examples: `tests/examples/test_examples.py`.
- Hardware-sensitive checks are gated with markers and skip logic from `tests/conftest.py`.

## Common Patterns

**Async Testing:**
```python
@pytest.mark.xdist_group("serial")
def test_thread_injection_detected(tmp_path):
    kernel = (
        "import threading\n"
        "import time\n"
        "\n"
        "def run(x, y):\n"
        "    threading.Thread(target=lambda: time.sleep(10), daemon=True).start()\n"
        "    return x + y\n"
    )
    traces = _run_eval_driver(tmp_path, kernel)
    assert traces[0]["evaluation"]["status"] == "REWARD_HACK"
```

**Error Testing:**
```python
with pytest.raises(ValidationError, match="HIP/C\\+\\+ and Python cannot be mixed"):
    _make_spec(languages=["pytorch", "hip_cpp"], entry_point="kernel.hip::run")
```

**Hardware Markers:**
- `requires_rocm`: requires a ROCm GPU visible through PyTorch.
- `requires_rocm_dev`: requires ROCm HIP development headers.
- `requires_ck`: requires Composable Kernel headers.
- `requires_rocwmma`: requires rocWMMA headers.
- `requires_rdna4`: requires AMD RDNA 4, such as `gfx1200`.
- `requires_cdna3`: requires AMD CDNA 3, such as `gfx942`.
- `requires_cutile`: legacy NVIDIA marker that is always skipped in this ROCm-only port.
- `timing_serial`: GPU timing tests skipped by default unless selected with `pytest tests -m timing_serial -n 0`.

---

*Testing analysis: 2026-05-24*
