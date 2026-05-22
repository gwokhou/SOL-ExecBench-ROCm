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

**Shipped version:** v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness, archived 2026-05-22.

The v1.0 milestone migrated the repository to a ROCm-only runtime baseline. The
v1.1 milestone added CDNA 3 code/schema support, maintained active
CUDA/NVIDIA-residue audit coverage, ROCm-facing public example paths, clarified
compatibility-example metadata, and a concrete CDNA 3 hardware validation
handoff for a future milestone. The v1.2 milestone harvested selected
engineering practices from `hip-execbench` as internal diagnostics, reporting,
score-interpretation guardrails, and public-contract tests without changing
external schemas, CLI behavior, trace formats, or examples. The v1.3 milestone
closed non-CDNA residual gaps by comparing the ROCm port
against NVIDIA SOL ExecBench's original public functionality and selectively
borrowing engineering practices from
`~/PyCharmMiscProject/hip-playground/hip-execbench`. The v1.4 milestone added a
compatibility inventory, internal evidence/report modeling, CDNA 3 validation
readiness metadata, and RDNA 4 validation evidence while still excluding real
CDNA 3 hardware validation and any CDNA 3 hardware-validation claim.

Validation status:

- RDNA 4 (`gfx1200`) full adapted suite passed: `462 passed, 58 skipped`.
- CDNA 3 (`gfx940`, `gfx941`, `gfx942`) is supported by schema/build/docs.
- CDNA 3 (`gfx94*`) full adapted suite validation remains deferred by user
  decision and must be completed before claiming CDNA 3 hardware validation.

## Current Milestone: v1.5 AMD-native SOL Scoring and ROCm Profiler Timing

**Goal:** Build a SOLAR-like AMD-native scoring foundation and replace default
ROCm timing with an accuracy-first profiler-backed timing model, while keeping
CDNA 3 real hardware validation explicitly out of scope.

**Target features:**

- Implement a SOLAR-like AMD pipeline for graph extraction, FLOP/byte analysis,
  hardware model input, and auditable SOL-bound output for ROCm targets.
- Promote SOL Score from formula/helper plus baseline comparison into an
  end-to-end AMD-native scoring workflow with explicit bound, baseline, score,
  suite aggregation, and claim guardrails.
- Replace the default timing path with a ROCm profiler-backed kernel activity
  timing backend where that is the most accurate path.
- Investigate Triton, HIP native, and PyTorch operator timing semantics
  separately, and expose source-specific timer backends when a unified timing
  backend would reduce accuracy.
- Preserve existing public contracts unless a scoring or timing extension
  requires an additive, documented output path.

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
- CDNA 3 `gfx94*` schema targets and HIP offload staging are supported without claiming hardware validation - v1.1.
- Active source, tests, docs, examples, scripts, Docker, README, and pyproject have maintained CUDA/NVIDIA residue audit coverage - v1.1.
- Public native examples use HIP-facing paths and solution filenames - v1.1.
- Former NVIDIA library/DSL examples use ROCm compatibility-example terminology and portable examples include CDNA 3 metadata where appropriate - v1.1.
- CDNA 3 hardware validation handoff is documented for the next milestone - v1.1.
- `hip-execbench` engineering practices have been assessed and mapped to
  accepted, rejected, and deferred SOL ExecBench ROCm adaptations - v1.2.
- Internal ROCm diagnostics and trace reporting helpers improve operator
  inspection without changing public CLI or trace contracts - v1.2.
- SOL-Score interpretation guardrails warn against unsupported AMD-native
  performance claims while preserving the existing score formula - v1.2.
- Public contract guardrail tests protect schemas, CLI help, trace behavior,
  HIP-facing examples, and CDNA 3 validation deferral language - v1.2.
- Original NVIDIA SOL ExecBench public functionality is mapped to ROCm port
  disposition with tests protecting parity documentation - v1.3.
- Public baseline comparison over existing trace JSONL is available through
  `sol-execbench-baseline` without changing trace schemas or the main benchmark
  CLI - v1.3.
- ROCm library categories distinguish supported HIP/C++ from candidate
  `hipblas`, `miopen`, `ck`, and `rocwmma` replacement directions - v1.3.
- `hip-execbench` baseline/reporting practices have been selectively adapted or
  explicitly rejected according to SOL ExecBench public-contract constraints -
  v1.3.
- Non-CDNA validation debt is closed; real CDNA 3 hardware validation remains
  the only deferred project-level item - v1.3.
- `hip-execbench` engineering practice adaptation, internal evidence/report
  modeling, CDNA 3 validation readiness metadata, and RDNA 4 validation
  evidence are implemented without changing public SOL ExecBench ROCm
  contracts - v1.4.

### Active

- [ ] Implement a SOLAR-like AMD-native SOL bound pipeline for ROCm targets.
- [ ] Implement an end-to-end AMD-native SOL scoring workflow with explicit
      guardrails and suite/workload aggregation.
- [ ] Replace default timing with an accuracy-first ROCm profiler-backed timing
      path.
- [ ] Define and expose Triton, HIP native, and PyTorch operator timing
      semantics separately when needed to preserve timing accuracy.

### Out of Scope

- Maintaining CUDA/NVIDIA runtime compatibility - ROCm is the target platform for this port.
- Preserving NVIDIA-specific dependency implementations when a ROCm replacement is required - equivalent behavior matters more than retaining original code.
- Guaranteeing one-to-one high-performance replacements for every NVIDIA DSL - some former CUTLASS/cuDNN/CuTe/cuTile categories remain documented compatibility examples or candidates.
- Claiming CDNA 3 hardware validation before a full `gfx94*` suite pass is recorded.
- Performing real CDNA 3 `gfx94*` full-suite hardware validation in v1.5 - user explicitly excluded it from this milestone.
- Claiming NVIDIA leaderboard equivalence or B200/SOLAR equivalence for AMD-native scores.

## Context

The current codebase is now ROCm-oriented. `docker/Dockerfile` uses a ROCm
development base image. Python dependency declarations point PyTorch and
torchvision at ROCm wheels. The package remains a Python CLI benchmark runner:
`src/sol_execbench/cli/main.py` loads problem JSON, `ProblemPackager` stages
files and injects AMD architecture flags, and `src/sol_execbench/driver/templates/`
contains the HIP-aware build and eval subprocess scripts.

High-risk areas for future work remain real CDNA 3 hardware validation, native
library replacement quality, timing integrity on additional AMD architectures,
and AMD-native score interpretation. v1.5 focuses on a SOLAR-like AMD-native
scoring model and profiler-backed timing. Timing accuracy is the highest
priority: Triton, HIP native, and PyTorch operator sources may require separate
timer semantics instead of one forced unified timer.

## Constraints

- **Platform**: ROCm >= 7.0 is the supported software baseline.
- **Hardware**: RDNA 4 is validated; CDNA 3 is code/schema-supported but hardware validation remains deferred.
- **Compatibility**: Preserve SOL ExecBench benchmark semantics and public schemas unless a ROCm-specific change is unavoidable.
- **Scope**: NVIDIA/CUDA runtime support is intentionally not maintained.
- **Licensing**: All retained and replacement code must comply with the repository LICENSE and third-party dependency obligations.
- **Quality**: Migrated tests, examples, Docker checks, and end-to-end evaluation must pass under ROCm.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| ROCm-only port | User explicitly chose to replace NVIDIA/CUDA paths rather than maintain dual backend compatibility. | Validated in v1.0 |
| Broad solution ecosystem target | User chose to pursue ROCm equivalents for all original solution categories, not just HIP-only or a minimal subset. | Partially validated in v1.1; some former NVIDIA DSL categories remain compatibility examples/documented candidates |
| Adapted test suite is the completion gate | User defined done as passing the migrated existing tests on ROCm, rather than adding a separate parity suite as v1 scope. | Validated on RDNA 4 in v1.0 |
| Preserve benchmark standards | The port should remain consistent with the NVIDIA SOL ExecBench paper and implementation standards wherever feasible. | Validated for public schemas and eval semantics in v1.0 |
| Defer CDNA 3 validation | No CDNA 3 hardware run was available during closure. | Accepted as v1.0 technical debt |
| Separate CDNA 3 code support from hardware validation | User requested CDNA 3 support in v1.1 while deferring real hardware validation to the next milestone. | Validated in v1.1 |
| Preserve public interfaces during practice harvest | User requested borrowing engineering experience from `hip-execbench` without changing this project's external interfaces or formats. | Validated in v1.2 |
| Close non-CDNA gaps before CDNA hardware validation | User requested v1.3 converge all issues except CDNA 3 validation, using NVIDIA SOL ExecBench parity and `hip-execbench` engineering experience as references. | Validated in v1.3 |
| Preserve compatibility during v1.4 engineering adaptation | User requested comprehensive `hip-execbench` engineering practice adaptation without breaking existing CLI, schema, trace, solution, or benchmark semantics. | Validated in v1.4 |
| Implement CDNA 3 readiness without hardware claim | User clarified v1.4 should implement CDNA 3 validation workflow readiness but not perform or claim real CDNA 3 validation. | Validated in v1.4 |
| RDNA 4 is the validation platform for v1.4 | User clarified v1.4 implementation must be validated on RDNA 4 with unit and E2E evidence. | Validated in v1.4 |
| Build a SOLAR-like AMD model | User selected a SOLAR-like pipeline over a minimal configuration-only roofline model for v1.5. | Pending in v1.5 |
| Replace default timing with profiler timing | User selected replacing the default event timing path with a profiler-backed path for v1.5. | Pending in v1.5 |
| Accuracy beats unified timing semantics | User clarified that Triton, HIP native, and PyTorch timing semantics must be investigated separately; if one unified timing definition is inaccurate, expose operator-source-specific timer backends. | Pending in v1.5 |
| Exclude CDNA 3 hardware validation from v1.5 | User explicitly said the next milestone does not include CDNA 3 validation. | Pending in v1.5 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-22 after starting v1.5 milestone*
