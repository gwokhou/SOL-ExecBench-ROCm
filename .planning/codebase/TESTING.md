# Testing Patterns

**Analysis Date:** 2026-05-28

## Test Framework

**Runner:**
- Pytest >=9.0.2 from `pyproject.toml`.
- Parallel execution uses `pytest-xdist` with `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`.
- Config: `pyproject.toml` and `tests/conftest.py`.

**Assertion Library:**
- Plain `assert` statements and `pytest.raises`.
- Click command tests use `click.testing.CliRunner` in `tests/sol_execbench/test_contract.py` and `tests/sol_execbench/test_cli_environment_snapshot.py`.

**Run Commands:**
```bash
uv run pytest tests/              # Run all tests with configured xdist defaults
uv run pytest tests/sol_execbench/test_e2e.py              # Run one test file
uv run pytest tests -m timing_serial -n 0                  # Run timing tests that are skipped by default
uv run --with ruff ruff check .                            # Lint source and tests
uv run ty check                                            # Type-check configured src/tests inputs
```

## Test File Organization

**Location:**
- Main package coverage lives under `tests/sol_execbench/`.
- Tests for nested source packages may mirror source directories: `tests/sol_execbench/core/bench/test_timing.py`, `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/driver/test_problem_packager.py`.
- Docker and environment dependency checks live under `tests/docker/dependencies/`.
- Example workflow coverage lives under `tests/examples/`.
- Shared type-safe model builders live in `tests/sol_execbench_type_helpers.py`.

**Naming:**
- Test files use `test_*.py`.
- Test functions use explicit behavior names like `test_cuda_cflags_rejected_with_hip_cflags_guidance` and `test_profile_collection_unavailable_is_nonfatal_metadata`.
- Group related tests in `Test*` classes when a module has many related behaviors: `TestLanguageValidation`, `TestEntryPointSuffixValidation`, `TestCompile`.

**Structure:**
```text
tests/
+-- conftest.py                         # ROCm markers, hardware skip policy, shared cache fixture
+-- sol_execbench_type_helpers.py       # Typed Pydantic model factories for tests
+-- sol_execbench/
|   +-- core/bench/test_*.py            # Package-local unit coverage
|   +-- core/data/test_*.py             # Schema and validation coverage
|   +-- driver/test_*.py                # Driver and staging coverage
|   +-- test_*.py                       # Cross-cutting, CLI, docs, scoring, migration coverage
+-- docker/dependencies/test_*.py       # Container/runtime dependency checks
+-- examples/test_examples.py           # Runnable example coverage
```

## Test Structure

**Suite Organization:**
```python
class TestLanguageValidation:
    @pytest.mark.parametrize("langs", [["pytorch"], ["triton"]])
    def test_pure_python_languages_accepted(self, langs):
        spec = _make_spec(languages=langs)
        assert set(spec.languages) == {SupportedLanguages(lg) for lg in langs}

    def test_legacy_cuda_nvidia_languages_rejected_with_guidance(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_spec(languages=["cuda_cpp"], entry_point="kernel.hip::run")
        assert "cuda_cpp" in str(exc_info.value)
```

**Patterns:**
- Keep module-level constants for reusable sample payloads: `_DEFINITION_DICT`, `_WORKLOAD_DICTS`, `_PYTHON_SOLUTION_DICT` in `tests/sol_execbench/driver/test_problem_packager.py`.
- Use small local helpers for repeated setup: `_make_spec` in `tests/sol_execbench/core/data/test_solution.py`, `_make_packager` in `tests/sol_execbench/driver/test_problem_packager.py`.
- Use `pytest.mark.parametrize` for schema matrices, target architecture matrices, and invalid input tables: `tests/sol_execbench/core/data/test_solution.py`.
- Use dataclasses for larger parametrized E2E descriptors: `Sample` and `EvilCase` in `tests/sol_execbench/test_e2e.py`.
- Mark serial GPU/E2E work with `pytest.mark.xdist_group("serial")` where needed: `tests/sol_execbench/test_e2e.py`.

## Mocking

**Framework:** Pytest fixtures, `monkeypatch`, injectable callables, and lightweight fake functions. `unittest.mock` appears in driver template tests where module execution is isolated.

**Patterns:**
```python
def runner(command: Sequence[str], cwd, timeout) -> subprocess.CompletedProcess[str]:
    calls.append(list(command))
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=0,
        stdout='{"definition": "demo"}\n',
        stderr="profiler note",
    )

result = collect_rocprofv3_profile(request, runner=runner)
assert calls[0][-3:] == ["--", "python", "eval_driver.py"]
```

**What to Mock:**
- Mock subprocess runners at injection points instead of invoking external tools: `collect_rocprofv3_profile(..., runner=runner)` in `tests/sol_execbench/test_rocm_profiler.py`.
- Mock environment variables with `monkeypatch.setenv` and `monkeypatch.delenv`: `tests/sol_execbench/test_cli_environment_snapshot.py`.
- Mock dataset downloads and CLI execution functions to test orchestration without network or full benchmark runs: `tests/sol_execbench/test_download_solexecbench.py`, `tests/sol_execbench/test_run_dataset_amd_score.py`.
- Mock filesystem probes with temp directories when testing diagnostics: `tests/sol_execbench/test_rocm_diagnostics_reporting.py`.

**What NOT to Mock:**
- Do not mock Pydantic validation when testing public schema behavior; instantiate real models through helpers in `tests/sol_execbench_type_helpers.py`.
- Do not mock staged file writes for `ProblemPackager`; assert real files in `tmp_path` as in `tests/sol_execbench/driver/test_problem_packager.py`.
- Do not run GPU or ROCm-dependent checks unguarded; use markers from `tests/conftest.py`.

## Fixtures and Factories

**Test Data:**
```python
def make_solution(**kwargs: Any) -> Solution:
    return Solution.model_validate(kwargs)

@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SOLEXECBENCH_CACHE_PATH", str(cache_dir))
    return cache_dir
```

**Location:**
- Typed factories: `tests/sol_execbench_type_helpers.py`.
- Shared pytest hooks and environment fixtures: `tests/conftest.py`.
- E2E sample problems: `tests/sol_execbench/samples/`.
- Example kernels and references: `examples/`.

## Coverage

**Requirements:** No explicit coverage percentage or coverage tool configuration is detected in `pyproject.toml`.

**View Coverage:**
```bash
Not configured
```

## Test Types

**Unit Tests:**
- Scope: Pydantic schema validation, scoring helpers, timing policy, profiler command construction, path safety, and pure utility behavior.
- Approach: instantiate real models, assert exact payload fields, and check errors with `pytest.raises`; examples include `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/test_rocm_profiler.py`, `tests/sol_execbench/core/bench/test_timing.py`.

**Integration Tests:**
- Scope: CLI sidecars, problem staging, run-dataset orchestration, Docker script policy, documentation guardrails, and ROCm dependency probes.
- Approach: use `tmp_path`, `CliRunner`, fake subprocess/dataset runners, and real JSON sidecar reads; examples include `tests/sol_execbench/test_cli_environment_snapshot.py`, `tests/sol_execbench/driver/test_problem_packager.py`, `tests/docker/dependencies/test_rocm_runtime.py`.

**E2E Tests:**
- Framework: Pytest.
- Scope: package sample problems, compile HIP/C++ solutions when needed, execute evaluation subprocesses, parse trace JSON, and assert `EvaluationStatus.PASSED`.
- Location: `tests/sol_execbench/test_e2e.py` and `tests/examples/test_examples.py`.
- Hardware-sensitive E2E tests must use the ROCm markers and skip policy from `tests/conftest.py`.

## Common Patterns

**Async Testing:**
```python
Not used; tests are synchronous and subprocess-oriented.
```

**Error Testing:**
```python
with pytest.raises(ValidationError, match="HIP/C\\+\\+ and Python cannot be mixed"):
    _make_spec(languages=["pytorch", "hip_cpp"], entry_point="kernel.hip::run")
```

**Hardware Markers:**
- Use `@pytest.mark.requires_rocm` for tests that need a ROCm GPU visible through PyTorch.
- Use `@pytest.mark.requires_rocm_dev` for HIP extension header/toolchain tests.
- Use `@pytest.mark.requires_ck` and `@pytest.mark.requires_rocwmma` for library header checks.
- Use `@pytest.mark.requires_rdna4` and `@pytest.mark.requires_cdna3` for architecture-specific tests.
- Use `@pytest.mark.timing_serial` for GPU timing tests; these are skipped by default unless `-m timing_serial` is selected.
- `requires_cutile` is registered as a legacy NVIDIA marker and skipped in this ROCm-only port.

---

*Testing analysis: 2026-05-28*
