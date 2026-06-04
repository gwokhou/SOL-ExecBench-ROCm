# Compliance And Known Gaps

## License

This repository is distributed under Apache-2.0. See `LICENSE`.

This repository is a ROCm port of NVIDIA SOL-ExecBench. The port retains files
derived from the original implementation and preserves NVIDIA copyright notices
where they apply to upstream-retained or derivative material. Independent ROCm
work uses project attribution. See `docs/provenance.md` and
`provenance.toml` for the active provenance policy.

NVIDIA notices identify retained upstream material. They do not imply NVIDIA
endorsement, NVIDIA runtime support, or NVIDIA ownership of independent ROCm
work added in this port. AMD product and ROCm references describe the target
runtime ecosystem and do not imply AMD endorsement.

The SOL-ExecBench paper is cited as benchmark and methodology context. Paper
citation is separate from file-level source copyright ownership.

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
| Ruff, Ty | Development-time linting, formatting, and type checking. |

Review each package's upstream license before redistributing binaries or
container images that include those dependencies.

## Attribution And Provenance

Active file provenance is classified as:

- upstream retained,
- derivative modified,
- independent ROCm work,
- generated or planning material.

Header handling follows `docs/provenance.md`. Prior blanket NVIDIA headers are
corrected through ordinary commits; this project does not rewrite git history
for ordinary metadata cleanup unless a separate legal review requires it.

## Dataset Redistribution

NVIDIA SOL-ExecBench evaluation dataset content is not redistributed by this
project. Original NVIDIA rows, definitions, workloads, traces, solutions, blobs,
and ROCm-migrated derivatives must be obtained and generated locally by users
with applicable rights under the NVIDIA Evaluation Dataset License. They must
not be committed to the repository or included in release/prerelease bundles.

FlashInfer Trace content is tracked separately as Apache-2.0 material from
`flashinfer-ai/flashinfer-trace`. Redistribution of FlashInfer Trace material
requires preserving Apache-2.0 license and attribution notices, and migration
metadata must not collapse FlashInfer Trace provenance into the NVIDIA
Evaluation Dataset License boundary.

The enforceable source and redistribution matrix lives in `provenance.toml`.
Run the CPU-safe guardrail before publishing or after staging dataset-related
changes:

```bash
uv run scripts/check_dataset_redistribution.py --staged
uv run scripts/check_dataset_redistribution.py --release-root out/prerelease_artifact_bundle
```

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
