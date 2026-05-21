# Phase 06 Context

## Goal

Make the ROCm port usable and legally clear for researchers and developers.

## Requirements

- SCFG-04: Documentation states that this port is ROCm-only and does not maintain CUDA/NVIDIA runtime support.
- DOC-01: README and setup docs explain ROCm installation, Docker usage, dataset setup, and local evaluation commands.
- DOC-02: Schema docs describe ROCm-supported languages, hardware targets, and replacement limitations.
- DOC-03: Profiling/analyze docs explain the ROCm-native tooling path.
- DOC-04: License and third-party notices are reviewed and updated for retained and replacement dependencies.
- DOC-05: Known gaps or unsupported NVIDIA-equivalent features are documented clearly.

## Existing State

- `README.md` still described NVIDIA Container Toolkit, B200, CUDA C++, CUTLASS, cuDNN, CuTe DSL, and cuTile as active support.
- `docs/solution.md` still documented CUDA/NVIDIA schema values that the ROCm schema now rejects.
- `docs/trace.md` used NVIDIA hardware examples and did not explain ROCm environment fields.
- `docs/definition.md` used CUDA wording for the custom input device parameter.
- No dedicated ROCm setup, profiling, compliance, or limitations document existed.

## Constraints

- Preserve the Apache-2.0 license text and existing NVIDIA copyright notices.
- Do not claim CDNA 3 support until deferred TEST-05 evidence is collected.
- Keep original benchmark citation and dataset attribution, while clearly marking this repository as a ROCm-only port.
