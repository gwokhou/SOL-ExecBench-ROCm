# Testing Patterns

**Analysis Date:** 2026-05-22

## Test Framework

**Runner:**
- Pytest `>=9.0.2` with `pytest-xdist>=3.5`, configured in `pyproject.toml`.
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`.
- Default addopts: `-n auto --dist loadgroup`, so tests run in parallel unless a command overrides xdist.

**Assertion Library:**
- Native `assert` statements plus `pytest.raises`, `pytest.mark.parametrize`, fixtures, and monkeypatching.
- `unittest.mock` is used where call ordering and patched subprocess behavior matter: `tests/sol_execbench/core/bench/test_clock_lock.py`, `tests/sol_execbench/driver/test_build_ext.py`.

**Run Commands:**
```bash
uv run pytest tests/                                      # Run all tests with default xdist settings
uv run pytest tests/sol_execbench/test_e2e.py             # Run one test file
uv run pytest tests -m timing_serial -n 0                 # Run GPU timing tests skipped by default
uv run pytest tests/sol_execbench/core/data/test_solution.py -n 0  # Run a focused file serially
```

## Test File Organization

**Location:**
- Package tests live under `tests/sol_execbench/` and usually mirror `src/sol_execbench/`: `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/core/bench/test_io.py`, `tests/sol_execbench/driver/test_problem_packager.py`.
- Example workflow coverage lives under `tests/examples/`, especially `tests/examples/test_examples.py`.
- Docker and environment dependency checks live under `tests/docker/dependencies/`.
- Malicious sample kernels used by reward-hack tests live under `tests/sol_execbench/samples/evil_*`.

**Naming:**
- Test modules use `test_*.py`.
- Test functions use descriptive `test_*` names that state the expected behavior: `test_legacy_cuda_nvidia_languages_rejected_with_guidance`, `test_live_collection_labels_failed_profiler_as_fallback`.
- Group related tests in `Test*` classes without inheritance: `TestLanguageValidation`, `TestCompileOptions`, `TestProbeClockLockAvailable`.

**Structure:**
```text
tests/
├── conftest.py                         # hardware marker registration, auto-skip logic, shared tmp cache fixture
├── sol_execbench/
│   ├── core/data/test_*.py             # schema/model validation
│   ├── core/bench/test_*.py            # timing, IO, correctness, clock lock, reward-hack logic
│   ├── driver/test_*.py                # packager and staged template behavior
│   └── test_*.py                       # e2e, ROCm migration audits, scoring/reporting behavior
├── examples/test_examples.py           # example directory end-to-end coverage
└── docker/dependencies/test_*.py        # ROCm container/toolchain checks
```

## Test Structure

**Suite Organization:**
```python
@pytest.mark.parametrize(("legacy", "replacement"), LEGACY_LANGUAGE_REPLACEMENTS)
def test_legacy_cuda_nvidia_languages_rejected_with_guidance(legacy, replacement):
    with pytest.raises(ValidationError) as exc_info:
        _make_spec(languages=[legacy], entry_point="kernel.hip::run")

    message = str(exc_info.value)
    assert legacy in message
    assert replacement in message
```

**Patterns:**
- Define module-level sample dictionaries for schema and packaging tests, then construct Pydantic/domain objects through fixtures: `tests/sol_execbench/driver/test_problem_packager.py`.
- Use `pytest.mark.parametrize` for matrix behavior across languages, hardware targets, suffixes, and examples: `tests/sol_execbench/core/data/test_solution.py`, `tests/examples/test_examples.py`.
- Use `tmp_path` for all generated files and staging directories. `ProblemPackager` tests write into `tmp_path / "staging"`.
- Use explicit failure messages when subprocess or e2e assertions fail, including stdout and stderr: `tests/sol_execbench/test_e2e.py`, `tests/examples/test_examples.py`.
- Use `pytest.importorskip()` for optional libraries in tests that can still run in reduced environments: `tests/sol_execbench/core/bench/test_io.py`.

## Mocking

**Framework:** Pytest `monkeypatch`, `unittest.mock.patch`, `MagicMock`, and injectable runner callables.

**Patterns:**
```python
def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    (tmp_path / "timing.csv").write_text(ROCPROFV3_CSV)
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=0,
        stdout="profiled",
        stderr="",
    )
```

```python
with patch("sol_execbench.core.bench.clock_lock.subprocess.run", return_value=result):
    assert verify_clocks(1, 1) is True
```

```python
monkeypatch.setattr(problem_packager.subprocess, "check_output", fake_check_output)
monkeypatch.setenv("SOLEXECBENCH_CACHE_PATH", str(cache_dir))
```

**What to Mock:**
- External commands and privileged ROCm tooling: `rocm-smi`, `rocminfo`, `rocm_agent_enumerator`, `rocprofv3`, `hipcc`.
- PyTorch extension compilation in template tests by replacing `torch.utils.cpp_extension` modules in `sys.modules`: `tests/sol_execbench/driver/test_build_ext.py`.
- GPU event/synchronization calls when a unit test does not need real hardware: `tests/sol_execbench/core/bench/test_timing.py`.
- Profiler execution through the `runner` parameter instead of patching `subprocess.run` directly: `tests/sol_execbench/test_rocm_profiler.py`.

**What NOT to Mock:**
- Pydantic schema validation for public data contracts; instantiate real models such as `BuildSpec`, `Solution`, `Definition`, and `Trace`.
- JSON serialization and staged file writes in `ProblemPackager`; assert real files under `tmp_path`.
- End-to-end subprocess execution in `tests/sol_execbench/test_e2e.py` and `tests/examples/test_examples.py` unless the test is explicitly a unit test for command construction.

## Fixtures and Factories

**Test Data:**
```python
@pytest.fixture
def config() -> BenchmarkConfig:
    return BenchmarkConfig(lock_clocks=False)

def _make_spec(**overrides):
    base = dict(
        languages=["triton"],
        target_hardware=["LOCAL"],
        entry_point="kernel.py::run",
    )
    base.update(overrides)
    return BuildSpec(**base)
```

**Location:**
- Shared infrastructure fixtures live in `tests/conftest.py`.
- Module-specific fixtures and factories live near the tests that use them: `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/core/data/test_solution.py`.
- Example and sample descriptors use dataclasses in their test modules: `Example` in `tests/examples/test_examples.py`, `Sample` and `EvilCase` in `tests/sol_execbench/test_e2e.py`.
- File fixtures are normally built inline using `tmp_path`; no central fixtures directory is detected.

## Coverage

**Requirements:** No coverage threshold or coverage tool configuration is detected in `pyproject.toml`.

**View Coverage:**
```bash
uv run pytest tests/        # Project does not configure a coverage command
```

## Test Types

**Unit Tests:**
- Schema/model validation tests cover Pydantic validators, enum values, dtype conversion, shape rules, and trace constraints: `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/core/data/test_definition.py`, `tests/sol_execbench/core/data/test_dtypes.py`.
- Core bench unit tests cover correctness, IO, timing helpers, reward-hack detection, and clock locking: `tests/sol_execbench/core/bench/`.
- Scoring and reporting unit tests cover AMD score/SOL behavior and guardrails: `tests/sol_execbench/test_amd_native_score.py`, `tests/sol_execbench/test_amd_sol_bounds.py`, `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py`.

**Integration Tests:**
- Driver/template tests stage real files and execute template code in controlled temp directories: `tests/sol_execbench/driver/test_problem_packager.py`, `tests/sol_execbench/driver/test_build_ext.py`, `tests/sol_execbench/driver/test_eval_driver.py`.
- E2E tests package samples/examples, optionally compile HIP/C++ kernels, run subprocess evaluation, parse trace JSONL, and assert all traces pass: `tests/sol_execbench/test_e2e.py`, `tests/examples/test_examples.py`.
- ROCm profiling tests cover command construction, CSV parsing, fallback behavior, and runner-driven live collection: `tests/sol_execbench/test_rocm_profiler.py`.

**E2E Tests:**
- Pytest-based, not a separate browser or external E2E framework.
- Example and sample E2E tests are xdist-serialized with `pytest.mark.xdist_group("serial")` for cases that run GPU/subprocess workflows.
- Hardware-sensitive tests use markers registered in `tests/conftest.py`: `requires_rocm`, `requires_rdna4`, `requires_cdna3`, `requires_cutile`, `timing_serial`, and `cpp`.

## Common Patterns

**Async Testing:**
```python
result = subprocess.run(
    cmd,
    cwd=pkg.output_dir,
    capture_output=True,
    text=True,
    timeout=300,
)
assert result.returncode == 0, (
    f"Execution failed:\nstdout={result.stdout}\nstderr={result.stderr}"
)
```
- No asyncio test framework is used. Long-running work is tested through bounded subprocess calls with `timeout`.

**Error Testing:**
```python
with pytest.raises(ValidationError, match="HIP/C\\+\\+ and Python cannot be mixed"):
    _make_spec(languages=["pytorch", "hip_cpp"], entry_point="kernel.hip::run")
```
- Match important message fragments for public validation guidance.
- For subprocess and profiler fallback behavior, assert structured fields such as `returncode`, `stderr`, `fallback_applied`, and `selection.reason` instead of only checking truthiness.

---

*Testing analysis: 2026-05-22*
