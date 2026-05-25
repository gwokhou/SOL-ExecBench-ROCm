# Phase 68 Context: External ROCm Toolchain Research

## Objective

Record primary-source ROCm toolchain research before implementing routing
logic. The phase focuses on tool capabilities, output types, lifecycle state,
repository migration status, and evidence-level boundaries.

## Inputs

- `.planning/research/ROCM_TOOLCHAIN_ROUTING.md`
- ROCprofiler-SDK `rocprofv3` documentation
- ROCm Systems and deprecated `rocprofiler-systems` repositories
- RGA repository and GPUOpen manual
- HIP compiler documentation
- LLVM `llvm-objdump` documentation

## Decisions

- Treat ROCm tool fragmentation as a first-class benchmark concern.
- Keep Static Kernel Evidence out of v1.16.
- Use primary-source references in registry entries and docs.
