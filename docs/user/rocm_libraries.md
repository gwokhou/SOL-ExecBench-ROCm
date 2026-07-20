# ROCm solution categories

The solution schema accepts these ROCm-facing language categories:

| Value | Build/runtime path | Current checked-in sample |
| --- | --- | --- |
| `pytorch` | staged Python | `tests/sol_execbench/samples/custom_inputs_matmul` |
| `triton` | staged Python with Triton ROCm | `tests/sol_execbench/samples/nemotron_rms_norm` |
| `hip_cpp` | PyTorch HIP/C++ extension | `tests/sol_execbench/samples/rmsnorm/solution_cuda.json` |
| `hipblas` | native HIP path with caller-declared library flags | none |
| `miopen` | native HIP path with caller-declared library flags | none |
| `ck` | native HIP/header path | none |
| `rocwmma` | native HIP/header path | none |

Schema acceptance and a shared build route do not prove that every library,
operation family or hardware architecture is installed or validated. In this
revision only the paths listed in the final column are checked-in runnable
samples. Library-specific submissions must declare their source, dependencies,
compile options and target hardware explicitly and are validated by the same
solution/build schemas.

Native entry points may use `.hip`, `.cpp`, `.cc`, `.cxx`, `.c`, `.h` and
`.hpp`. Python/Triton and native categories cannot be mixed in one solution.
CUDA/NVIDIA language values and unsafe compiler/linker path injection are
rejected with ROCm migration guidance.

Typical external development dependencies are:

| Category | Representative header/library |
| --- | --- |
| hipBLAS | `hipblas/hipblas.h`, `libhipblas.so` |
| MIOpen | `miopen/miopen.h`, `libMIOpen.so` |
| Composable Kernel | CK headers supplied by the installed ROCm/library stack |
| rocWMMA | `rocwmma/rocwmma.hpp` |

Missing external libraries are environment/toolchain failures, not evidence
that the schema or all devices support the category. Hardware support claims
require focused compile, correctness and timing tests on the named architecture.
