# Phase 02 Pattern Map - ROCm Schema and Native Build Layer

## PATTERN MAPPING COMPLETE

## Target Files and Closest Analogs

| Target | Role | Closest analog / pattern |
|--------|------|--------------------------|
| `src/sol_execbench/core/data/solution.py` | Pydantic schema and enum source of truth | Existing `SupportedLanguages`, `SupportedHardware`, `CompileOptions`, and `BuildSpec._validate_languages` in the same file |
| `src/sol_execbench/driver/problem_packager.py` | Staging and target flag injection | Existing `_get_local_sm`, `_sm_to_gencode`, `_inject_gencode_flags`, and `_is_cpp` flow |
| `src/sol_execbench/driver/templates/build_ext.py` | Native build subprocess template | Existing template-level `Solution` validation, source discovery, `ext.load`, and `.so` rename logic |
| `tests/sol_execbench/core/data/test_solution.py` | Schema validation tests | Existing parametrized language category and suffix tests |
| `tests/sol_execbench/driver/test_problem_packager.py` | Packager staging/flag tests | Existing tests around injected `cuda_cflags` and target hardware |
| `tests/sol_execbench/driver/test_build_ext.py` | Template tests with mocked PyTorch extension loader | Existing `_exec_build_ext` helper and `mock.load.call_args` assertions |
| `tests/sol_execbench/test_rocm_schema_build_audit.py` | New static audit guard | Existing Docker dependency residue tests in Phase 1 show static grep-style gates; implement as Python pytest for precise allowlist reasons |

## Reusable Patterns

- Pydantic model validators are already used for entry-point and language support matrix checks.
- Tests prefer explicit `pytest.mark.parametrize` tables for language and suffix combinations.
- Build-template tests execute the template text in a temporary working directory with a mocked `torch.utils.cpp_extension` module, avoiding real compiler requirements.
- `ProblemPackager.compile()` writes normalized/injected `solution.json` before returning the compile command; tests can inspect the staged JSON.
- Static residue gates should be path-scoped and deterministic.

## Integration Notes

- `src/sol_execbench/__init__.py`, `src/sol_execbench/core/__init__.py`, and `src/sol_execbench/core/data/__init__.py` re-export schema classes and should only need changes if class names change. `CompileOptions` remains the same class name.
- `src/sol_execbench/driver/templates/eval_driver.py` still references native CUDA language values. That is intentionally Phase 3 unless Phase 2 schema changes require a narrow compatibility edit.
- `docs/user/solution.md`, examples, and broad example tests still contain NVIDIA/CUDA names. They are outside the failing Phase 2 audit scope.

## Suggested Plan Slices

1. Schema enums and compile-option model changes.
2. ProblemPackager AMD gfx detection and `--offload-arch` injection.
3. Build template HIP/C++ source discovery and extension-loader argument adaptation.
4. Focused audit guard and final coverage tests.
