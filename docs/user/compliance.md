# Compliance And Known Gaps

## License

This repository is distributed under Apache-2.0. See `LICENSE`.

This repository is a ROCm GPU-kernel benchmark whose problem corpus is derived
from AMD AgentKernelArena (AKA, Apache-2.0). The SOL-ExecBench paper is cited
as methodology context for the construction and evaluation framework; the
benchmark problems themselves are AKA-native AMD operators. Files derived from
AKA preserve AMD-AGI copyright notices where they apply to upstream-retained or
derivative material. Independent ROCm work uses project attribution. See
`docs/user/provenance.md` and `provenance.toml` for the active provenance
policy.

Upstream AKA notices identify retained upstream material. They do not imply
AMD-AGI endorsement, AMD runtime support, or AMD ownership of independent ROCm
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
| ROCm runtime and tools | HIP compiler, `rocminfo`, `amd-smi`, `rocprofv3`, AMD GPU runtime libraries. |
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

Header handling follows `docs/user/provenance.md`. Prior blanket NVIDIA headers are
corrected through ordinary commits; this project does not rewrite git history
for ordinary metadata cleanup unless a separate legal review requires it.

## Dataset Redistribution

The problem corpus is derived from AMD AgentKernelArena under Apache-2.0 and is
redistributable: authored definitions, workloads, and references under
`problems/RX_9060_XT/**` may be committed and included in release bundles
provided the Apache-2.0 license text and AMD-AGI/AgentKernelArena attribution
are preserved. The local AKA source clone under `data/AgentKernelArena/` is a
gitignored build input (covered by the `/data/*` ignore rule) and is not
committed.

FlashInfer Trace content is tracked separately as Apache-2.0 material from
`flashinfer-ai/flashinfer-trace`. Redistribution of FlashInfer Trace material
requires preserving Apache-2.0 license and attribution notices, and its
provenance must not be collapsed into any other source boundary.

The enforceable source and redistribution matrix lives in `provenance.toml`.
Run the CPU-safe guardrail before publishing or after staging dataset-related
changes:

```bash
uv run scripts/check_dataset_redistribution.py --staged
uv run scripts/check_dataset_redistribution.py --release-root out/prerelease_artifact_bundle
```

## Runtime Boundary

This distribution supports the ROCm language, library, container, profiling,
and hardware values enumerated by the current strict schemas. Values outside
those closed enums are rejected without migration aliases.

## Known Gaps

- CDNA 3 MI308X (`gfx942`) validation infrastructure evidence exists: the
  adapted pytest suite passed and a full dataset validation run completed with
  documented timeout blockers and expected Quant NVFP4/MXFP4 skips. Do not
  claim full hardware validation for the MI300X GPU model under the CDNA 3
  family until those blockers and required benchmark-grade exact-hardware
  MI300X evidence are resolved or explicitly bounded.
- NVFP4/MXFP4 Quant benchmark ROCm adaptation and hardware validation are
  deferred because no CDNA4-class hardware is currently available. CDNA3 runs
  should report these workloads as expected hardware-unsupported skips rather
  than replacing benchmark reference semantics with portable dequantized
  fallbacks.
- Explicit schema hardware values include `gfx1200`, `gfx940`, `gfx941`,
  `gfx942`, and `LOCAL`. The `gfx94*` entries are code/schema support by
  themselves; hardware-validation claims require archived real-hardware
  evidence and accepted failure/skip boundaries.
- Original SOL-Score references a NVIDIA B200 roofline model. Scores computed
  from that model are not an AMD hardware roofline claim.
- Some examples remain PyTorch compatibility examples for former NVIDIA
  library/DSL categories until ROCm-native library variants are implemented and
  validated.
