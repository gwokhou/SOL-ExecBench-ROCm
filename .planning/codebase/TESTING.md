# Testing Patterns

**Analysis Date:** 2026-05-26

## Test Framework

**Runner:**
- Pytest `>=9.0.2`, configured in `pyproject.toml`.
- Pytest-xdist `>=3.5`, configured with default `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`.
- Config: `pyproject.toml` and runtime marker logic in `tests/conftest.py`.

**Assertion Library:**
- Plain Python `assert` is the default assertion style across `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/core/bench/test_clock_lock.py`, and `tests/examples/test_examples.py`.
- Use `pytest.raises()` for error behavior, as in `tests/sol_execbench/core/data/test_solution.py` and `tests/sol_execbench/test_rocm_diagnostics_reporting.py`.
- Use `pytest.approx()` for numeric tolerance assertions, as in `tests/sol_execbench/core/bench/test_correctness.py`.
- Use `torch.testing.assert_close()` for tensor comparisons, as in `tests/docker/dependencies/test_pytorch_rocm.py`.

**Run Commands:**
```bash
uv run pytest tests/                         # Run the full suite with configured xdist defaults
uv run pytest tests/sol_execbench/test_e2e.py # Run one focused test file
uv run pytest tests/sol_execbench/core/data/ # Run schema-focused tests
uv run pytest tests/sol_execbench/driver/    # Run driver-focused tests
uv run pytest tests/examples/test_examples.py -k consistency # Run example file consistency checks
uv run pytest tests -m timing_serial -n 0    # Run GPU timing tests that are skipped by default
uv run pytest tests/docker/dependencies/     # Run ROCm container dependency checks
uv run ruff check .                          # Run lint checks used by CI
uv run ty check                              # Run type checks used by CI
```

## Test File Organization

**Location:**
- Package tests live under `tests/sol_execbench/`, mirroring source areas such as `tests/sol_execbench/core/data/`, `tests/sol_execbench/core/bench/`, and `tests/sol_execbench/driver/`.
- Example workflow tests live under `tests/examples/`, especially `tests/examples/test_examples.py`.
- Docker and ROCm dependency readiness checks live under `tests/docker/dependencies/`.
- Some legacy or source-tree tests are present under mirrored package paths such as `tests/sol_execbench/core/bench/test_io.py`; new tests should follow the nearest existing target area under `tests/sol_execbench/`.

**Naming:**
- Use `test_*.py` files and `test_*` functions, as in `tests/sol_execbench/test_baseline_comparison.py`.
- Use descriptive behavior names such as `test_legacy_cuda_nvidia_languages_rejected_with_guidance()` in `tests/sol_execbench/core/data/test_solution.py`.
- Group related cases in `PascalCase` test classes when the file covers many behaviors, such as `TestLockClocks` and `TestVerifyClocks` in `tests/sol_execbench/core/bench/test_clock_lock.py`.

**Structure:**
```text
tests/
├── conftest.py                         # Marker registration, ROCm skip logic, shared tmp_cache_dir
├── sol_execbench_type_helpers.py        # Typed model factory helpers
├── sol_execbench/
│   ├── core/data/test_*.py              # Pydantic schema and contract tests
│   ├── core/bench/test_*.py             # IO, timing, correctness, reward-hack, clock tests
│   ├── driver/test_*.py                 # Staging/build/eval-driver behavior
│   ├── samples/                         # Self-contained e2e sample inputs and malicious kernels
│   └── test_*.py                        # CLI, docs, migration, scoring, and integration guardrails
├── examples/test_examples.py            # Runnable examples plus consistency checks
└── docker/dependencies/test_*.py         # ROCm runtime/container dependency checks
```

## Test Structure

**Suite Organization:**
```python
class TestLanguageValidation:
    @pytest.mark.parametrize("langs", [["pytorch"], ["triton"]])
    def test_pure_python_languages_accepted(self, langs):
        spec = _make_spec(languages=langs)
        assert set(spec.languages) == {SupportedLanguages(lg) for lg in langs}
```

This pattern appears in `tests/sol_execbench/core/data/test_solution.py`: module-level constants define the test matrix, small helpers build valid defaults, and classes group related validation behavior.

**Patterns:**
- Build valid default payloads in `_make_*` helpers and override one concern per test, as in `_make_spec()` in `tests/sol_execbench/core/data/test_solution.py`.
- Use module-level dictionaries for schema fixtures, as in `_DEFINITION_DICT`, `_WORKLOAD_DICTS`, and `_HIP_SOLUTION_DICT` in `tests/sol_execbench/driver/test_problem_packager.py`.
- Use dataclass case descriptors for larger e2e matrices, such as `Example` in `tests/examples/test_examples.py` and `Sample` in `tests/sol_execbench/test_e2e.py`.
- Keep subprocess e2e assertions explicit and include stdout/stderr in failure messages, as in `test_example()` in `tests/examples/test_examples.py`.
- Use `pytest.param(..., id=..., marks=...)` for parametrized cases that need stable IDs and marker control, as in `tests/examples/test_examples.py`.

## Mocking

**Framework:** Pytest `monkeypatch`, `unittest.mock`, and injected runner callables.

**Patterns:**
```python
_MODULE = "sol_execbench.core.bench.clock_lock"

with patch(f"{_MODULE}.subprocess.run", return_value=probe_result) as mock_run:
    result = probe_clock_lock_available()
```

Use this import-path patching style for subprocess and timing dependencies, as in `tests/sol_execbench/core/bench/test_clock_lock.py`.

```python
def runner(command: Sequence[str], cwd, timeout) -> subprocess.CompletedProcess[str]:
    calls.append(list(command))
    return subprocess.CompletedProcess(args=list(command), returncode=0, stdout="", stderr="")
```

Use injected runner functions for command-building modules that already accept dependency injection, as in `tests/sol_execbench/test_rocm_profiler.py`.

**What to Mock:**
- Mock subprocess calls and system tools such as `rocm-smi`, `rocprofv3`, and dataset download helpers; examples are in `tests/sol_execbench/core/bench/test_clock_lock.py`, `tests/sol_execbench/test_rocm_profiler.py`, and `tests/sol_execbench/test_download_solexecbench.py`.
- Mock environment variables with `monkeypatch.setenv()` and `monkeypatch.delenv()`, as in `tests/conftest.py`, `tests/sol_execbench/core/bench/test_clock_lock.py`, and `tests/sol_execbench/test_cli_environment_snapshot.py`.
- Mock expensive CLI or dataset execution in orchestration tests, as in `tests/sol_execbench/test_run_dataset_execution_closure.py` and `tests/sol_execbench/test_run_dataset_amd_score.py`.

**What NOT to Mock:**
- Do not mock Pydantic validation for public schemas; instantiate real models through `tests/sol_execbench_type_helpers.py`.
- Do not mock example JSON files for consistency checks; `tests/examples/test_examples.py` reads actual `examples/` files and verifies source content matches JSON payloads.
- Do not mock ROCm hardware in dependency readiness tests under `tests/docker/dependencies/`; those tests assert real PyTorch ROCm and toolchain availability.
- Avoid mocking trace contracts when testing CLI output; `tests/sol_execbench/test_baseline_comparison.py` loads real `Trace` models from temporary JSONL files.

## Fixtures and Factories

**Test Data:**
```python
def make_solution(**kwargs: Any) -> Solution:
    return Solution.model_validate(kwargs)
```

Factory helpers in `tests/sol_execbench_type_helpers.py` wrap Pydantic `model_validate()` for `Definition`, `Workload`, `Solution`, `BuildSpec`, and `Trace`.

**Location:**
- Shared factories: `tests/sol_execbench_type_helpers.py`.
- Global pytest fixtures and marker skip logic: `tests/conftest.py`.
- Driver fixture payloads: `tests/sol_execbench/driver/test_problem_packager.py`.
- SOLAR derivation fixtures: `tests/sol_execbench/solar_derivation_fixtures.py`.
- E2E samples: `tests/sol_execbench/samples/` and `examples/`.

## Coverage

**Requirements:** No line, branch, function, or statement coverage threshold is configured. `docs/TESTING.md` states there is no coverage threshold in `pyproject.toml` and no coverage config file such as `.coveragerc`.

**View Coverage:**
```bash
# Not configured. Add pytest-cov explicitly before requesting coverage reports.
uv run pytest tests/
```

CI coverage is behavioral rather than percentage-based: `.github/workflows/code-quality.yml` runs Ruff, Ty, CPU-safe `tests/sol_execbench`, and example consistency checks across Python 3.12 and 3.13.

## Test Types

**Unit Tests:**
- Schema validation tests under `tests/sol_execbench/core/data/` cover Pydantic models in `src/sol_execbench/core/data/`, including ROCm language and hardware validation in `tests/sol_execbench/core/data/test_solution.py`.
- Bench utility tests under `tests/sol_execbench/core/bench/` cover timing, correctness, IO, reward-hack detection, and clock locking for modules in `src/sol_execbench/core/bench/`.
- Scoring tests under `tests/sol_execbench/test_amd_sol_v2.py`, `tests/sol_execbench/test_amd_native_score.py`, and `tests/sol_execbench/test_solar_derivation_evidence.py` verify deterministic scoring and evidence semantics.

**Integration Tests:**
- Driver tests under `tests/sol_execbench/driver/` validate staging, generated build scripts, and evaluation driver behavior for `src/sol_execbench/driver/`.
- CLI tests use `click.testing.CliRunner`, as in `tests/sol_execbench/test_baseline_comparison.py`, `tests/sol_execbench/test_contract.py`, and `tests/sol_execbench/test_cli_environment_snapshot.py`.
- Dataset and script tests under `tests/sol_execbench/test_dataset_inventory_readiness.py`, `tests/sol_execbench/test_run_dataset_execution_closure.py`, and `tests/sol_execbench/test_parity_gap_report.py` exercise filesystem outputs with `tmp_path`.

**E2E Tests:**
- `tests/sol_execbench/test_e2e.py` packages self-contained samples, runs compile/execute subprocess phases, and checks all traces pass.
- `tests/examples/test_examples.py` runs examples from `examples/` and checks source-file consistency against JSON payloads.
- Hardware-sensitive examples are marked with `cpp`, `requires_rocm`, `requires_rocm_dev`, `requires_ck`, `requires_rocwmma`, `requires_rdna4`, or `requires_cdna3` through `tests/conftest.py` and parametrized marks in `tests/examples/test_examples.py`.

## Markers and Hardware Gates

**Configured Markers:**
- `cpp`: HIP/C++ extension compilation; declared in `pyproject.toml`.
- `timing_serial`: GPU timing tests skipped by default unless selected with `-m timing_serial`; registered in `tests/conftest.py`.
- `requires_rocm`: requires visible ROCm GPU through PyTorch; checked in `tests/conftest.py`.
- `requires_rocm_dev`: requires HIP development headers under `/opt/rocm`; checked in `tests/conftest.py`.
- `requires_ck`: requires Composable Kernel headers under `/opt/rocm/include/ck/ck.hpp`; checked in `tests/conftest.py`.
- `requires_rocwmma`: requires rocWMMA headers under `/opt/rocm/include/rocwmma/rocwmma.hpp`; checked in `tests/conftest.py`.
- `requires_rdna4` and `requires_cdna3`: require AMD `gfx12*` or `gfx94*` architectures; checked in `tests/conftest.py`.
- `requires_cutile`: legacy NVIDIA marker that is always skipped in this ROCm-only port; checked in `tests/conftest.py`.

## Common Patterns

**Async Testing:**
```python
# Not used. Tests are synchronous and use subprocess.CompletedProcess or injected runners.
```

**Error Testing:**
```python
with pytest.raises(ValidationError, match="require a .py entry point"):
    _make_spec(languages=["triton"], entry_point="kernel.hip::run")
```

Use `pytest.raises(..., match=...)` for schema, diagnostics, and guardrail failures, as in `tests/sol_execbench/core/data/test_solution.py` and `tests/sol_execbench/test_solar_derivation_evidence.py`.

**Temporary Files:**
```python
candidate = tmp_path / "candidate.jsonl"
candidate.write_text(...)
```

Use `tmp_path` for JSONL traces, staged packages, generated sidecars, and script outputs, as in `tests/sol_execbench/test_baseline_comparison.py`, `tests/sol_execbench/driver/test_problem_packager.py`, and `tests/sol_execbench/test_dataset_inventory_readiness.py`.

**CLI Testing:**
```python
result = CliRunner().invoke(cli, ["--candidate", str(candidate), "--baseline", str(baseline)])
assert result.exit_code == 0
```

Use `CliRunner` for Click entry points in `src/sol_execbench/cli/`, as in `tests/sol_execbench/test_baseline_comparison.py`.

---

*Testing analysis: 2026-05-26*
