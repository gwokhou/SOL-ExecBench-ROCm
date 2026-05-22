# Project Research: Architecture for v1.8

## Integration Points

### Solution Schema

`src/sol_execbench/core/data/solution.py` already exposes `miopen`, `ck`, and
`rocwmma` as native/C++ solution languages. v1.8 should preserve this public
contract and focus on making those values runnable and tested.

### Native Packaging

`ProblemPackager` treats all three categories as C++ languages and compiles them
through `build_ext.py`. The architecture should keep that flow:

1. Solution JSON declares one library category.
2. Source files are staged into a temporary build directory.
3. `compile_options` provide include, HIP, and linker flags.
4. `build_ext.py` compiles `benchmark_kernel.so`.
5. `eval_driver.py` imports the compiled extension and runs the entry point.

### Examples

Each library category should get a focused example under `examples/`:

- `examples/miopen/softmax/`
- `examples/ck/gemm/` or `examples/ck/gemm_bias_relu/`
- `examples/rocwmma/gemm/`

These examples should follow the same four-file public pattern:

- `definition.json`
- `reference.py`
- `workload.jsonl`
- `solution_<category>.json`

### Tests

Tests should be layered:

- Schema/build tests verify staging and compile metadata without requiring a
  real GPU.
- Dependency tests verify MIOpen, CK, and rocWMMA headers/libraries are present
  in the Docker/ROCm environment.
- Example tests run on RDNA 4 using existing `requires_rocm`/architecture
  markers.
- Documentation tests protect support-status wording and deferred CDNA 3/CDNA 4
  claims.

## Suggested Build Order

1. Build plumbing and dependency diagnostics for the three categories.
2. MIOpen softmax replacement because it maps directly to an existing former
   cuDNN compatibility example.
3. CK GEMM or fused GEMM replacement because it maps to the former CUTLASS
   compatibility example and CK's documented strengths.
4. rocWMMA GEMM replacement with RDNA 4 support constraints.
5. Compatibility cleanup and RDNA 4 validation closure.

## Data Flow

The data flow remains unchanged from the benchmark perspective:

`definition/workload/solution -> ProblemPackager -> native extension compile -> eval_driver -> Trace JSONL`

Library-specific code lives inside solution examples and compile metadata, not
inside canonical trace or schema models.
