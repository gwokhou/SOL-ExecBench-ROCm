# Testing Patterns

**Analysis Date:** 2026-06-04

## Test Framework

**Runner:**
- Pytest `>=9.0.2` from the `dev` dependency group in `pyproject.toml`.
- Pytest-xdist `>=3.5` is enabled by default.
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`.

**Assertion Library:**
- Native `assert` statements and `pytest.raises`.
- Pydantic validation tests assert `ValidationError` from `pydantic`, as in `tests/sol_execbench/core/data/test_solution.py`.

**Run Commands:**
```bash
uv run pytest tests/                         # Run the full suite with configured xdist
uv run pytest tests/sol_execbench/test_e2e.py # Run one test module
uv run pytest tests -m timing_serial -n 0     # Run serial GPU timing tests normally skipped by conftest
uv run pytest -m requires_rocm -q -rs         # Run ROCm hardware tests and show skip reasons
uv run pytest tests/examples/test_rocm_cli_paths.py -q -rs # Run example ROCm CLI coverage
uv run ty check                               # Type-check src and tests per pyproject.toml
```

## Test File Organization

**Location:**
- Package and feature tests live under `tests/sol_execbench/`.
- Tests mirroring package subtrees live under paths such as `tests/sol_execbench/core/data/`, `tests/sol_execbench/core/bench/`, and `tests/sol_execbench/driver/`.
- Example workflow tests live under `tests/examples/`.
- Container dependency smoke tests live under `tests/docker/dependencies/`.
- Test fixtures and sample problems live under `tests/sol_execbench/fixtures/`, `tests/sol_execbench/samples/`, and `tests/samples/`.

**Naming:**
- Use `test_*.py` for test files.
- Use descriptive test functions that state expected behavior: `test_dangerous_native_compile_options_rejected`, `test_noisy_user_stdout_stays_out_of_trace_jsonl`, and `test_profile_collection_unavailable_is_nonfatal_metadata`.
- Use test classes to group schema behavior without shared mutable state, as in `TestLanguageValidation`, `TestEntryPointSuffixValidation`, and `TestHardwareAndCompileOptions` in `tests/sol_execbench/core/data/test_solution.py`.

**Structure:**
```text
tests/
├── conftest.py                         # Custom ROCm markers and shared tmp cache fixture
├── sol_execbench_type_helpers.py        # Typed Pydantic factory helpers
├── sol_execbench/
│   ├── core/data/test_*.py              # Schema and data contract tests
│   ├── core/bench/test_*.py             # Runtime, timing, reward-hack, and IO tests
│   ├── driver/test_*.py                 # Generated driver and packager tests
│   ├── test_*.py                        # CLI, docs, dataset, scoring, matrix, and evidence tests
│   ├── fixtures/                        # JSON fixture payloads
│   └── samples/                         # Sample problem directories and malicious kernels
├── examples/test_*.py                   # Example workflow tests
└── docker/dependencies/test_*.py        # Container dependency checks
```

## Test Structure

**Suite Organization:**
```python
@pytest.mark.parametrize(("field", "flag", "message"), [
    ("hip_cflags", "@/tmp/flags.rsp", "response file"),
    ("cflags", "-I/usr/local/include", "host paths"),
])
def test_dangerous_native_compile_options_rejected(field, flag, message):
    with pytest.raises(ValidationError, match=message):
        _make_spec(
            languages=["hip_cpp"],
            entry_point="kernel.hip::run",
            compile_options={field: [flag]},
        )
```

**Patterns:**
- Put small local factory helpers near tests when data setup is feature-specific. Examples: `_make_spec` in `tests/sol_execbench/core/data/test_solution.py`, `_run_eval_driver_process` in `tests/sol_execbench/driver/test_eval_driver.py`, and `_matching_default_dependency_env` in `tests/sol_execbench/test_run_docker_dependency_preflight.py`.
- Put reusable typed Pydantic factories in `tests/sol_execbench_type_helpers.py` (`make_solution`, `make_build_spec`, `make_trace`, `json_dict`, `typed`).
- Use `tmp_path` for all generated files, staged problems, output bundles, and sidecars. Avoid writing into repo paths during tests unless the test is explicitly reading committed fixtures.
- Assert full structured payloads or critical fields, not just truthiness. Examples: profiler metadata assertions in `tests/sol_execbench/test_rocm_profiler.py` and docker preflight JSON assertions in `tests/sol_execbench/test_run_docker_dependency_preflight.py`.
- For scripts and CLIs, assert both exit code and absence/presence of dangerous command text. `tests/sol_execbench/test_run_docker_dependency_preflight.py` checks that blocked preflight paths do not print `docker build` or `docker run`.

## Mocking

**Framework:** Pytest `monkeypatch`, local fake classes, and injected runner callables. `unittest.mock` is not a dominant pattern.

**Patterns:**
```python
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
```

**What to Mock:**
- Mock external command runners for profiler and toolchain logic. Use runner injection as in `tests/sol_execbench/test_rocm_profiler.py`.
- Monkeypatch hardware detection helpers in `tests/conftest.py` when testing marker behavior, as in `tests/sol_execbench/test_cdna3_hardware_marker.py`.
- Monkeypatch environment variables for Docker and timing policy tests, as in `tests/sol_execbench/test_run_docker_dependency_preflight.py` and `tests/sol_execbench/core/bench/test_make_eval_clock_warn.py`.
- Use fake pytest config/item classes for collection hook tests instead of invoking pytest subprocesses. See `_FakeConfig` and `_FakeItem` in `tests/sol_execbench/test_cdna3_hardware_marker.py`.

**What NOT to Mock:**
- Do not mock Pydantic validation when testing schemas; instantiate real models through `model_validate` or typed helpers in `tests/sol_execbench_type_helpers.py`.
- Do not mock generated driver syntax; parse the actual template with `ast.parse` in `tests/sol_execbench/driver/test_eval_driver.py`.
- Do not mock JSONL trace parsing or staged file layouts in subprocess integration tests; write real `definition.json`, `workload.jsonl`, `solution.json`, and source files under `tmp_path`.

## Fixtures and Factories

**Test Data:**
```python
def make_solution(**kwargs: Any) -> Solution:
    return Solution.model_validate(kwargs)
```

**Location:**
- `tests/sol_execbench_type_helpers.py` contains reusable typed factories for Pydantic models.
- `tests/conftest.py` contains shared pytest hooks and `tmp_cache_dir`, which sets `SOLEXECBENCH_CACHE_PATH` to `tmp_path / "cache"`.
- Static JSON fixtures live under `tests/sol_execbench/fixtures/solar_derivation/`.
- Sample problem directories live under `tests/sol_execbench/samples/` and `tests/samples/`.
- Inline minimal problem payloads are acceptable for local subprocess tests, as shown by `_MINIMAL_DEFINITION`, `_MINIMAL_WORKLOAD`, and `_SOLUTION_SPEC` in `tests/sol_execbench/driver/test_eval_driver.py`.

## Coverage

**Requirements:** No numeric coverage target is enforced in `pyproject.toml`.

**View Coverage:**
```bash
uv run pytest tests/ # Primary verification command; no coverage plugin is configured
```

## Test Types

**Unit Tests:**
- Schema and contract tests validate Pydantic fields, validators, enum values, and JSON schema behavior under `tests/sol_execbench/core/data/`.
- Pure logic tests validate scoring, compatibility matrices, dataset layout/readiness, sharding, trust summaries, and evidence models under `tests/sol_execbench/test_*.py`.
- Profiler command construction and CSV parsing tests live in `tests/sol_execbench/test_rocm_profiler.py`.

**Integration Tests:**
- Generated eval-driver subprocess tests live in `tests/sol_execbench/driver/test_eval_driver.py`; they write staging files, run `eval_driver.py` with `subprocess.run`, and parse canonical trace JSONL.
- E2E problem/sample tests live in `tests/sol_execbench/test_e2e.py` and use `ProblemPackager` plus subprocesses.
- CLI and example ROCm tests live in `tests/examples/test_rocm_cli_paths.py`.
- Dataset, artifact bundle, release validation, and reporting script tests live under `tests/sol_execbench/test_*script*.py`, `tests/sol_execbench/test_*report*.py`, and `tests/sol_execbench/test_prerelease_artifact_bundle.py`.

**E2E Tests:**
- Pytest is the E2E runner; no separate browser or service E2E framework is used.
- Use `pytest.mark.xdist_group("serial")` for tests that should not run concurrently because they use GPU state, subprocess staging, timing state, or global monkey-patch surfaces.
- Use `pytest.mark.requires_rocm` for live ROCm GPU tests and pair with architecture markers when needed.

## Markers

- `requires_rocm`: tests require a ROCm GPU visible through PyTorch. Registered in `pyproject.toml` and `tests/conftest.py`.
- `requires_rdna4`: tests require AMD RDNA 4 (`gfx12*`) hardware such as `gfx1200`.
- `requires_cdna3`: tests require AMD CDNA 3 (`gfx94*`) hardware such as `gfx942`.
- `requires_rocm_dev`: tests require ROCm HIP development headers under `/opt/rocm/include/hip/hip_runtime_api.h`; registered in `tests/conftest.py`.
- `requires_ck`: tests require Composable Kernel headers under `/opt/rocm/include/ck/ck.hpp`; registered in `tests/conftest.py`.
- `requires_rocwmma`: tests require rocWMMA headers under `/opt/rocm/include/rocwmma/rocwmma.hpp`; registered in `tests/conftest.py`.
- `requires_cutile`: legacy NVIDIA cuTile marker; `tests/conftest.py` always skips it in this ROCm-only port.
- `timing_serial`: GPU timing tests skipped by default unless selected with `-m timing_serial`; run with `-n 0`.
- `xdist_group("serial")`: use for tests that need serialized execution under the default `-n auto --dist loadgroup`.

## GPU, Container, and Profiler Patterns

- Hardware availability is centralized in `tests/conftest.py`. Do not duplicate ROCm detection in individual tests unless testing detection itself.
- `tests/conftest.py` first checks Linux device nodes `/dev/kfd` and `/dev/dri`, then checks `torch.version.hip`, `torch.cuda.is_available()`, and `gcnArchName`/`gfx_arch_name`.
- Live dependency smoke tests under `tests/docker/dependencies/` assert tool presence and behavior with real subprocesses. For example, `tests/docker/dependencies/test_rocm_runtime.py` checks `rocminfo`, `hipcc`, `rocprofv3`, and `amd-smi`/`rocm-smi`.
- Docker script policy tests use dry-run environment variables and assert JSON preflight policy without running Docker, as in `tests/sol_execbench/test_run_docker_dependency_preflight.py`.
- Profiler tests distinguish diagnostic profile artifacts from score-authoritative timing evidence. `tests/sol_execbench/test_rocm_profiler.py` asserts `diagnostic_only`, `score_authority`, `fallback_applied`, `backend`, `kernel_duration_ms`, and artifact kind classification.
- For `rocprofv3`, build commands with an explicit `--` separator before the application command. `test_build_rocprofv3_command_places_application_after_separator` in `tests/sol_execbench/test_rocm_profiler.py` is the pattern to preserve.

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
```

**Error Testing:**
```python
with pytest.raises(ValidationError, match="cuda_cflags"):
    _make_spec(
        languages=["hip_cpp"],
        entry_point="kernel.hip::run",
        compile_options={"cuda_cflags": ["-O3"]},
    )
```

**Subprocess Output Testing:**
```python
completed = _run_docker_dependency_preflight("--preflight-only", **env)
assert completed.returncode != 0
payload = json.loads(completed.stdout)
assert payload["benchmark_allowed"] is False
assert "docker run" not in completed.stdout
```

**Trace JSONL Testing:**
```python
traces = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
assert traces[0]["evaluation"]["status"] == "PASSED"
assert "run noise from solution" not in result.stdout
assert "run noise from solution" in result.stderr
```

---

*Testing analysis: 2026-06-04*
