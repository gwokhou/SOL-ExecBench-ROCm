# Testing Patterns

**Analysis Date:** 2026-06-04

## Test Framework

**Runner:**
- Pytest 9.x, configured in `pyproject.toml`.
- Default options: `-n auto --dist loadgroup`, so the suite uses
  `pytest-xdist` by default.
- Custom collection behavior lives in `tests/conftest.py`.

**Assertion Library:**
- Pytest built-in assertions.
- `pytest.raises(..., match=...)` is the standard error assertion pattern.
- `click.testing.CliRunner` is used for CLI tests.
- Tests assert Pydantic `ValidationError` for schema failures.

**Run Commands:**
```bash
uv run pytest tests/                                      # Run full test suite
uv run pytest tests/sol_execbench/test_e2e.py             # Run one test file
uv run pytest tests -m requires_rocm                      # Run ROCm-marked tests
uv run pytest tests -m requires_cdna3                     # Run CDNA3-gated tests
uv run pytest tests -m timing_serial -n 0                 # Run serial timing tests
uv run --with ruff ruff check .                           # Lint source and tests
uv run --with ruff ruff format .                          # Format Python files
```

## Test File Organization

**Location:**
- Tests live in `tests/`.
- Package tests live under `tests/sol_execbench/`.
- Tests for subpackages mirror the source tree when useful, for example
  `tests/sol_execbench/core/data/`, `tests/sol_execbench/core/bench/`, and
  `tests/sol_execbench/driver/`.
- Example workflow tests live under `tests/examples/`.
- Docker dependency checks live under `tests/docker/dependencies/`.

**Naming:**
- Test files use `test_*.py`.
- Test functions use descriptive `test_*` names that describe the expected
  behavior or regression.
- Test classes group related behavior, for example `TestLanguageValidation`,
  `TestEntryPointSuffixValidation`, and `TestHardwareAndCompileOptions`.

**Structure:**
```text
tests/
  conftest.py
  sol_execbench/
    core/
      data/
        test_definition.py
        test_solution.py
        test_workload.py
      bench/
        test_eval_runtime.py
        test_timing.py
        test_correctness.py
    driver/
      test_problem_packager.py
      test_build_ext.py
      test_eval_driver.py
    fixtures/
      solar_derivation/*.json
    samples/
      <sample_problem>/
        definition.json
        workload.jsonl
        solution_*.json
  examples/
    test_examples.py
    test_rocm_cli_paths.py
  docker/
    dependencies/
      test_rocm_runtime.py
      test_pytorch_rocm.py
```

## Test Structure

**Suite Organization:**
```python
@pytest.mark.parametrize("target", ["LOCAL", "gfx1200", "gfx940", "gfx941", "gfx942"])
def test_rocm_hardware_targets_accepted(target):
    spec = _make_spec(target_hardware=[target])
    assert spec.target_hardware == [SupportedHardware(target)]


def test_load_staged_problem_rejects_missing_definition(tmp_path):
    with pytest.raises(RuntimeError, match="definition.json not found"):
        load_staged_problem(tmp_path)
```

**Patterns:**
- Arrange/act/assert is used implicitly with small local helpers.
- Parametrization is common for schema matrices, language/category mappings,
  hardware targets, compile flag rejection, and docs guardrails.
- Helper factory functions reduce noise, such as `_make_spec`, `_solution`,
  `_load_sample`, and `_sample_definition_workload_trace`.
- Tests prefer exact assertions on returned models, JSON keys, sidecar payloads,
  and error text.
- Integration tests include detailed failure messages with captured stdout and
  stderr when subprocesses fail.

## Mocking

**Framework:**
- Pytest fixtures, `monkeypatch`, local fake callables, and direct temporary
  module replacement are the dominant mocking tools.
- The suite does not rely on a separate mocking framework by default.

**Patterns:**
```python
def test_measure_reference_latency_returns_failure_message_on_exception():
    def fake_time_fn(*args, **kwargs):
        raise RuntimeError("boom")

    result = measure_reference_latency(
        lambda x: x,
        [1],
        "cpu",
        warmup=1,
        rep=2,
        time_fn=fake_time_fn,
    )

    assert result.failure == "Reference timing failed: boom"
```

```python
def test_load_user_function_ignores_existing_simple_module_collision(tmp_path):
    collision = types.ModuleType("kernel")
    previous = sys.modules.get("kernel")
    sys.modules["kernel"] = collision
    try:
        fn = load_user_function(_solution(), tmp_path)
    finally:
        if previous is None:
            sys.modules.pop("kernel", None)
        else:
            sys.modules["kernel"] = previous
```

**What to Mock:**
- Timing functions, subprocess boundaries, filesystem roots, environment
  variables, hardware probes, imported modules, and optional dependency
  availability.
- Use dependency injection parameters when modules provide them, such as
  `time_fn` or `path_exists`.
- Use `tmp_path` for filesystem artifacts and `monkeypatch` for environment
  changes such as `SOLEXECBENCH_CACHE_PATH`.

**What NOT to Mock:**
- Pydantic schema validation logic.
- Pure scoring/reporting helpers.
- Public JSON contract serialization.
- Internal modules in integration tests where the goal is to validate staged
  evaluation, packaging, or CLI behavior.

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
    return make_build_spec(**base)
```

```python
@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SOLEXECBENCH_CACHE_PATH", str(cache_dir))
    return cache_dir
```

**Location:**
- Shared type/model factories live in `tests/sol_execbench_type_helpers.py`.
- Reusable JSON fixtures live under `tests/sol_execbench/fixtures/`.
- Self-contained benchmark samples live under `tests/sol_execbench/samples/`.
- Example parity assets live under `examples/` and are tested from
  `tests/examples/`.

## Coverage

**Requirements:**
- No numeric coverage threshold is configured in `pyproject.toml`.
- Coverage expectations are behavior-driven and contract-driven: schema
  contracts, ROCm migration guardrails, public docs wording, subprocess
  evaluation, dataset reporting, and hardware gating must keep passing.
- New schema or driver logic should get small unit tests.
- Changes to subprocess evaluation, GPU execution, Docker behavior, dataset
  execution, or scoring/report authority should add integration or guardrail
  coverage.

**Configuration:**
- Pytest config lives in `pyproject.toml`.
- Hardware and timing selection is handled in `tests/conftest.py`.
- Ruff excludes `examples` and `data`; tests still cover examples explicitly
  where needed.

**View Coverage:**
```bash
uv run pytest tests/                                      # Current configured test run
```

No coverage-report command is configured in the project metadata as of
2026-06-04.

## Test Types

**Unit Tests:**
- Validate Pydantic schemas, enum values, helper functions, pure scoring logic,
  reporting summaries, diagnostics, and guardrail predicates.
- Typically run CPU-safe and use direct object construction.
- Examples include `tests/sol_execbench/core/data/test_solution.py`,
  `tests/sol_execbench/core/bench/test_eval_runtime.py`, and
  `tests/sol_execbench/test_amd_hardware_models.py`.

**Integration Tests:**
- Validate packaging, generated evaluation drivers, subprocess execution,
  dataset runner behavior, Docker wrapper semantics, and CLI reports.
- Use `tmp_path`, staged sample problems, captured subprocess output, and JSONL
  trace conversion.
- Examples include `tests/sol_execbench/test_e2e.py`,
  `tests/sol_execbench/driver/test_problem_packager.py`, and
  `tests/sol_execbench/test_run_dataset_execution_closure.py`.

**Docs and Contract Tests:**
- Many tests assert public documentation, migration residue, release readiness,
  claim boundaries, support matrices, and generated report schemas.
- These tests are part of the quality strategy, not incidental snapshots.
- Examples include
  `tests/sol_execbench/test_rocm_migration_residue_audit.py`,
  `tests/sol_execbench/test_public_contract_guardrails.py`,
  `tests/sol_execbench/test_rocm_support_docs.py`, and
  `tests/sol_execbench/test_public_prerelease_docs.py`.

**Hardware-Gated Tests:**
- `requires_rocm` marks tests requiring a ROCm GPU visible through PyTorch.
- `requires_rdna4` targets AMD RDNA 4, such as `gfx1200`.
- `requires_cdna3` targets AMD CDNA 3, such as `gfx942`.
- `requires_rocm_dev`, `requires_ck`, and `requires_rocwmma` gate native
  development headers and ROCm library headers.
- `requires_cutile` is legacy NVIDIA-only and is skipped in this ROCm-only port.
- `timing_serial` is skipped by default unless selected with `-m timing_serial`
  and should run with `-n 0`.

## Common Patterns

**Async Testing:**
- The repository is mostly synchronous Python. Async-specific test patterns are
  not a prominent convention.

**Error Testing:**
```python
with pytest.raises(ValidationError, match="HIP/C\\+\\+ and Python cannot be mixed"):
    _make_spec(languages=["pytorch", "hip_cpp"], entry_point="kernel.hip::run")
```

```python
with pytest.raises(RuntimeError, match="benchmark_kernel.so not found"):
    load_user_function(solution, tmp_path)
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

**CLI Testing:**
- Use `CliRunner` for direct Click CLI invocation when full subprocess isolation
  is not required.
- Use real subprocess execution for staged benchmark driver behavior.

**Snapshot Testing:**
- No snapshot-testing framework is configured.
- Equivalent coverage is provided by explicit JSON key assertions, exact schema
  field checks, document text guardrails, and report payload comparisons.

---

*Testing analysis: 2026-06-04*
*Update when test patterns change*
