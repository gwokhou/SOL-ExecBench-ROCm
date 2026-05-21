# SOL ExecBench ROCm Port

## What This Is

SOL ExecBench ROCm Port is a ROCm-only fork of SOL ExecBench for evaluating
LLM-generated GPU kernels on AMD GPUs. It keeps the original benchmark
semantics and public contracts where practical while replacing CUDA/NVIDIA
runtime, build, timing, environment, example, and documentation paths with ROCm
equivalents.

## Core Value

Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm
hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## Current State

**Shipped version:** v1.0 ROCm Port, archived 2026-05-21.

The v1.0 milestone migrated the repository to a ROCm-only runtime baseline:
ROCm Docker and dependency configuration, HIP/C++ native build staging,
ROCm-native solution schema values, ROCm-compatible evaluation timing, AMD
hardware/environment reporting, migrated examples, adapted pytest semantics,
and user-facing setup/schema/analysis/compliance documentation.

Validation status:

- RDNA 4 (`gfx1200`) full adapted suite passed: `462 passed, 58 skipped`.
- CDNA 3 (`gfx94*`) full adapted suite validation is deferred to a later
  milestone and must be completed before claiming CDNA 3 support.

## Requirements

### Validated

- Existing CLI can load definitions, workloads, solutions, and emit trace results through `sol-execbench` - existing.
- Existing driver stages problem files, compiles native solutions, and runs evaluation in a subprocess - existing.
- Existing Pydantic schemas define public `definition.json`, `workload.jsonl`, `solution.json`, and trace contracts - existing.
- Existing benchmark helpers cover input generation, correctness checks, timing, clock checks, environment capture, and reward-hack detection - existing.
- ROCm >= 7.0 Docker/runtime baseline with HIP compiler tooling, ROCm profiling tools, PyTorch ROCm, Triton ROCm, and selected ROCm library smoke coverage - v1.0.
- ROCm-only solution metadata and schema validation with HIP/C++, AMD gfx targets, `hip_cflags`, and explicit rejection guidance for legacy CUDA/NVIDIA values - v1.0.
- HIP/C++ solution sources can be staged, compiled, and loaded through the existing driver flow with AMD offload architecture handling - v1.0.
- PyTorch ROCm, Triton ROCm, and HIP/C++ solutions run through the isolated eval driver and produce valid trace JSONL - v1.0.
- Benchmark timing no longer depends on CUPTI or CUDA-only tooling and uses ROCm-compatible PyTorch device events - v1.0.
- Environment and clock checks use AMD/ROCm tooling and report ROCm/HIP/PyTorch hardware context - v1.0.
- Public examples and sample categories are migrated to ROCm-compatible execution, documented replacements, or explicit fallback notes - v1.0.
- Adapted pytest markers and tests use ROCm/AMD semantics, with reward-hack defenses still active - v1.0.
- Documentation explains ROCm setup, solution schema, trace schema, analysis workflow, compliance, unsupported NVIDIA runtime features, and known gaps - v1.0.

### Active

- [ ] Complete CDNA 3 (`gfx94*`) full adapted suite validation and update hardware support evidence.
- [ ] Decide whether to add explicit CDNA 3 schema hardware values after validation evidence exists.
- [ ] Replace remaining PyTorch fallback examples for former NVIDIA library/DSL categories with validated ROCm-native implementations where performance matters.
- [ ] Define AMD-native scoring or roofline interpretation before making AMD hardware performance claims from SOL-Score-style outputs.

### Out of Scope

- Maintaining CUDA/NVIDIA runtime compatibility - ROCm is the target platform for this port.
- Preserving NVIDIA-specific dependency implementations when a ROCm replacement is required - equivalent behavior matters more than retaining original code.
- Guaranteeing one-to-one high-performance replacements for every NVIDIA DSL in v1.0 - some former CUTLASS/cuDNN/CuTe/cuTile categories remain documented fallbacks or candidates.
- Claiming CDNA 3 support before a full `gfx94*` suite pass is recorded.

## Next Milestone Goals

The next milestone should begin with `$gsd-new-milestone` and fresh
requirements. Recommended seed goals:

- Run and record CDNA 3 full-suite validation.
- Add CDNA 3 schema/support documentation only after validation.
- Prioritize ROCm-native replacements for any fallback examples that matter for benchmark quality.
- Clarify AMD-native scoring semantics for performance reports.

## Context

The current codebase is now ROCm-oriented. `docker/Dockerfile` uses a ROCm
development base image. Python dependency declarations point PyTorch and
torchvision at ROCm wheels. The package remains a Python CLI benchmark runner:
`src/sol_execbench/cli/main.py` loads problem JSON, `ProblemPackager` stages
files and injects AMD architecture flags, and `src/sol_execbench/driver/templates/`
contains the HIP-aware build and eval subprocess scripts.

High-risk areas for future work remain hardware validation breadth, native
library replacement quality, timing integrity on additional AMD architectures,
and score interpretation.

## Constraints

- **Platform**: ROCm >= 7.0 is the supported software baseline.
- **Hardware**: RDNA 4 is validated for v1.0; CDNA 3 validation remains deferred.
- **Compatibility**: Preserve SOL ExecBench benchmark semantics and public schemas unless a ROCm-specific change is unavoidable.
- **Scope**: NVIDIA/CUDA runtime support is intentionally not maintained.
- **Licensing**: All retained and replacement code must comply with the repository LICENSE and third-party dependency obligations.
- **Quality**: Migrated tests, examples, Docker checks, and end-to-end evaluation must pass under ROCm.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| ROCm-only port | User explicitly chose to replace NVIDIA/CUDA paths rather than maintain dual backend compatibility. | Validated in v1.0 |
| Broad solution ecosystem target | User chose to pursue ROCm equivalents for all original solution categories, not just HIP-only or a minimal subset. | Partially validated in v1.0; some former NVIDIA DSL categories remain fallback/documented candidates |
| Adapted test suite is the completion gate | User defined done as passing the migrated existing tests on ROCm, rather than adding a separate parity suite as v1 scope. | Validated on RDNA 4 in v1.0 |
| Preserve benchmark standards | The port should remain consistent with the NVIDIA SOL ExecBench paper and implementation standards wherever feasible. | Validated for public schemas and eval semantics in v1.0 |
| Defer CDNA 3 validation | No CDNA 3 hardware run was available during closure. | Accepted as v1.0 technical debt |

## Evolution

This document evolves at phase transitions and milestone boundaries.

---
*Last updated: 2026-05-21 after v1.0 milestone*
