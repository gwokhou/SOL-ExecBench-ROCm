# Project Research: Feature Expectations for v1.8

**Milestone:** v1.8 ROCm Library Ecosystem Completion

## Table Stakes

### Library Category Support

- Each remaining candidate category (`miopen`, `ck`, `rocwmma`) has at least
  one public runnable example.
- Each example uses the declared library category in `solution_*.json`.
- Each example compiles through the existing native packaging flow.
- Each example runs through `sol-execbench` and produces `PASSED` trace JSONL
  on RDNA 4.
- Each category has dependency smoke tests that fail with clear messages when
  ROCm packages are missing.

### Documentation

- `docs/rocm_libraries.md` distinguishes supported, guarded, and deferred
  categories based on runnable evidence.
- README and compliance docs no longer imply MIOpen, CK, or rocWMMA are merely
  future directions once examples and tests exist.
- Former NVIDIA category docs explain the exact ROCm replacement path.

### Compatibility Cleanup

- Former cuDNN softmax compatibility example becomes a MIOpen-backed example,
  or the compatibility-only path is explicitly retired.
- Former CUTLASS GEMM compatibility example becomes CK, rocWMMA, or hipBLAS
  backed, with each replacement documented.
- Former CuTe/cuTile Jamba projection examples are either replaced with a real
  ROCm native library path or kept as explicitly non-supported historical
  compatibility examples.

### Validation

- v1.8 completion evidence is RDNA 4 only.
- CDNA 3 and CDNA 4 are documented as deferred validation targets.
- Public-contract tests prevent unsupported hardware validation claims.

## Differentiators

- A support matrix that ties each original NVIDIA category to a concrete ROCm
  replacement, example path, tests, and validation status.
- Library examples chosen to represent useful SOL ExecBench operation classes:
  softmax for MIOpen, GEMM/fused epilogue for CK, and matrix-core GEMM for
  rocWMMA.
- A repeatable RDNA 4 validation checklist for library examples and dependency
  checks.

## Anti-Features

- Broad performance claims without profiler-backed evidence.
- Silent fallback from a declared library category to PyTorch compatibility
  code.
- Treating schema recognition as support without runnable example evidence.
- Expanding scope into full CDNA 3/CDNA 4 validation during v1.8.
