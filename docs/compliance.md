# Compliance And Known Gaps

## License

This repository is distributed under Apache-2.0. See `LICENSE`.

The port retains files derived from the original SOL ExecBench implementation
and preserves existing SPDX headers and NVIDIA copyright notices where present.
Those notices identify retained upstream material; they do not imply that this
ROCm port supports NVIDIA runtime execution.

## Third-Party Dependencies

Runtime and development dependencies are declared in `pyproject.toml` and
locked in `uv.lock`. Important dependency families include:

| Dependency family or configuration | Purpose |
| --- | --- |
| PyTorch ROCm / torchvision ROCm | Tensor runtime, reference execution, HIP-backed extension builds. |
| Triton ROCm | Triton kernel examples and evaluation. |
| ROCm runtime and tools | HIP compiler, `rocminfo`, `rocm-smi`, `rocprofv3`, AMD GPU runtime libraries. |
| Pydantic, Click, Rich | Schemas and CLI. |
| safetensors, datasets | Benchmark input loading and dataset workflows. |
| pytest, pytest-xdist | Test execution and parallelization tooling. |
| `tool.ruff` configuration | Linting and formatting policy declared in `pyproject.toml`; Ruff itself is not declared as a package dependency in this checkout. |

Review each package's upstream license before redistributing binaries or
container images that include those dependencies.

## Unsupported NVIDIA Runtime Features

This port is ROCm-only and does not support:

- CUDA runtime execution,
- NVIDIA Container Toolkit requirements,
- `nvidia-smi` clock management,
- CUPTI timing,
- `cuda_cpp` solution metadata,
- `cuda_cflags` compile options,
- B200 as an active hardware target,
- CUTLASS, cuDNN, cuBLAS, CuTe DSL, or cuTile as active NVIDIA runtimes.

Legacy dataset/example names may still contain historical directory names such
as `cuda_cpp`, `cudnn`, or `cute_dsl`. In this port, those examples are either
migrated to ROCm metadata/source or documented as PyTorch fallbacks.

## ROCm Replacement Notes

Replacement candidates documented during the port:

- CUDA C++ -> HIP/C++ (`hip_cpp`)
- cuBLAS -> hipBLAS or hipBLASLt
- cuDNN -> MIOpen or HIP/Triton kernels
- CUTLASS/CuTe/cuTile -> Composable Kernel, rocWMMA, HIP kernels, or Triton ROCm
- CUB/Thrust-style primitives -> hipCUB, rocPRIM, rocThrust

Library-specific replacements should be treated as candidates until compiled,
tested, and validated on the relevant hardware class.

## Known Gaps

- CDNA 3 full-suite validation is deferred. Do not claim CDNA 3 support until
  the adapted test suite passes on `gfx94*`.
- Explicit schema hardware values include `gfx1200`, `gfx940`, `gfx941`,
  `gfx942`, and `LOCAL`. The `gfx94*` entries are code/schema support, not
  hardware-validation evidence.
- Original SOL-Score references a NVIDIA B200 roofline model. Scores computed
  from that model are not an AMD hardware roofline claim.
- Some examples remain PyTorch compatibility examples for former NVIDIA
  library/DSL categories until ROCm-native library variants are implemented and
  validated.
