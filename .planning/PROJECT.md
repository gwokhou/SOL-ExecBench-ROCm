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

**Shipped version:** v1.12 Evaluator Contract Metadata and Boundary Guardrails,
completed 2026-05-25.

**Current milestone:** v1.13 ROCm Runtime Evidence and Environment Diagnostics.

**Latest milestone outcome:** the ROCm port now exposes GPU-free evaluator
contract metadata through `sol-execbench contract --json`, keeps contract-only
metadata out of canonical trace JSONL, and records SOL/HIP ownership boundary
guardrails for external consumers.

**Current milestone goal:** make benchmark runs reproducible and diagnosable by
attaching optional ROCm/GPU environment evidence and fast preflight diagnostics
without changing SOL ExecBench correctness, timing, scoring, or v1.12 consumer
defaults.

**Next planned milestone:** v1.14 Optional rocprofv3 Profiling Evidence, scoped
to opt-in profiler artifacts and command provenance after the v1.13 runtime
evidence surface is stable.

The v1.0 milestone migrated the repository to a ROCm-only runtime baseline.
Milestones v1.1-v1.6 added CDNA 3 code/schema support, maintained residue
audits, harvested selected `hip-execbench` practices, closed non-CDNA parity
gaps, added source-specific timing semantics, `rocprofv3` evidence helpers, AMD
SOL bound artifacts, derived AMD-native score reports, live profiler collection,
dataset score reporting, and compatibility/claim guardrails. The v1.7 milestone
added optimized scoring baseline artifacts, source-specific profiler evidence
collection, expanded reward-hack defenses, a runnable hipBLAS public example,
and MI300X validation-readiness guardrails.
The v1.8 milestone completed the remaining ROCm library ecosystem replacements
for RDNA 4 scope by adding MIOpen, CK, and rocWMMA public examples and claim
guardrails.
The v1.9 milestone completed the AMD SOL/SOLAR bound modeling pipeline for RDNA
4 scope by adding strict hardware model artifacts, a structured bound graph IR,
auditable operator estimates, v2 sidecars, coverage semantics, score/dataset
integration, documentation, and claim guardrails.
The v1.10 milestone completed the scoped paper-aligned automatic SOLAR
derivation system by adding fixture contracts, semantic provenance,
family-specific formula/byte/bound evidence, degraded complex-family evidence,
coverage/status sidecars, score guards, dataset-runner report integration, and
claim guardrails.
The v1.11 milestone completed the paper dataset parity inventory and bounded
execution-closure workflow by adding dataset acquisition/layout manifests,
inventory/readiness/ready-subset sidecars, execution-closure reports,
parity-gap reports, release-closure docs, and claim guardrails.
The v1.12 milestone retroactively recorded the evaluator contract metadata and
boundary guardrail work merged through PR #1, including `sol-execbench contract
--json`, SOL-owned compatibility metadata, and HIP-side consumer boundaries.
The v1.13 milestone starts from ROCm Systems/GPUOpen enhancement research and
focuses only on backward-compatible runtime environment evidence, diagnostic
commands, and architecture-aware smoke/preflight checks. `rocprofv3` artifact
collection is deferred to the planned v1.14 milestone, and static RGA/GPUOpen
ISA analysis remains later candidate work.

## Current Milestone: v1.13 ROCm Runtime Evidence and Environment Diagnostics

**Goal:** Make benchmark runs reproducible and diagnosable by attaching ROCm/GPU
environment evidence and fast preflight diagnostics without changing SOL
ExecBench correctness, timing, scoring, or v1.12 consumer defaults.

**Target features:**
- Add optional run-level environment snapshots for ROCm version, GPU identity,
  `gfx` target, visible devices, PyTorch ROCm metadata, and collection
  provenance.
- Add a standalone diagnostic CLI such as `sol-execbench doctor` or
  `sol-execbench env-snapshot`.
- Add lightweight ROCm/HIP preflight checks with architecture-aware skip
  behavior for RDNA 4 and CDNA 3.
- Preserve evaluator contract version `1.0`, primary CLI defaults, canonical
  trace semantics, and scoring/correctness behavior.

**Explicitly deferred:**
- Optional `rocprofv3` profile artifact collection to v1.14.
- RGA/code-object/static ISA analysis to a later static kernel evidence
  milestone.
- Contract version bump or consumer-side requirement that confirmed benchmark
  claims must include environment evidence.

Validation status:

- RDNA 4 (`gfx1200`) full adapted suite passed: `462 passed, 58 skipped`.
- CDNA 3 (`gfx940`, `gfx941`, `gfx942`) is supported by schema/build/docs.
- CDNA 3 (`gfx94*`) full adapted suite validation remains deferred by user
  decision and must be completed before claiming CDNA 3 hardware validation.

## Recently Shipped Milestone: v1.10 Paper-Aligned SOLAR Automatic Derivation

**Goal:** Upgrade the AMD SOL/SOLAR derived bound pipeline into a
paper-aligned automatic SOLAR derivation system that can extract richer
workload graphs, model currently unsupported operation families, produce
auditable hardware-bound evidence, and protect score reports from partial
coverage misuse.

**Target features:**
- Extend BoundGraph/SOLAR IR so attention, MoE, convolution, SSM/Mamba,
  embedding/positional, and linear projection families have explicit extraction
  and estimation paths instead of taxonomy-only placeholders.
- Derive FLOP, byte, movement, intermediate, confidence, and rationale evidence
  from reference/workload structure with deterministic unsupported/inexact
  degradation.
- Strengthen AMD SOL v2 sidecars and AMD-native score reports so complete,
  degraded, and unscored states are machine-verifiable.
- Preserve ROCm-only scope, canonical trace JSONL, public schemas, and primary
  CLI behavior.

**Explicitly deferred:**
- Original-paper 124-model / 235-problem extraction and curation pipeline.
- MI300X/CDNA 3 and CDNA 4 real-hardware validation.
- NVFP4/MXFP4 hardware validation.
- Official hosted leaderboard or submission service.

## Recently Shipped Milestone: v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure

**Goal:** Turn the paper's public benchmark dataset into a concrete,
auditable ROCm-port surface: acquired or layout-verified, counted, classified,
diagnosed, runnable in small ready subsets, and reported with explicit parity
and closure status.

**Target features:**
- Define and enforce the dataset acquisition/layout contract for the public
  SOL-ExecBench benchmark categories: `L1`, `L2`, `Quant`, and
  `FlashInfer-Bench`.
- Generate a machine-readable paper parity inventory with problem counts,
  categories, dtype coverage, forward/backward indicators, custom input and
  safetensors usage, reference availability, and solution availability.
- Classify each problem's ROCm readiness as ready, schema/input blocked, dtype
  blocked, custom-input blocked, runtime blocked, unsupported NVIDIA-only path,
  or needing hardware evidence.
- Extend the dataset runner and reports so ready problems can run in small
  batches and emit canonical traces, summary JSON, AMD-native score reports,
  AMD SOL/SOLAR sidecars, and SOLAR derivation sidecars.
- Produce a parity gap report with claim guardrails that distinguish inventory
  completion from full 235-problem ROCm validation or leaderboard equivalence.

**Explicitly deferred:**
- Recreating the original 124-model / 7,400-subgraph extraction and curation
  pipeline.
- Full upstream NVlabs/SOLAR equivalence.
- CDNA 3 / MI300X, CDNA 4, and NVFP4/MXFP4 real-hardware validation.
- Hosted leaderboard or remote submission service.
- Full 235-problem real-hardware validation unless this milestone discovers
  the environment and runtime budget are sufficient.

## Recently Shipped Milestone: v1.9 AMD SOL/SOLAR Bound Modeling Completion

**Goal:** Make the ROCm port's AMD SOL bound pipeline credible, structured,
auditable, and broad enough to support paper-style SOL scoring on the current
ROCm path, with validation scoped to RDNA 4 only.

**Target features:**
- Upgrade the current conservative AST-only estimator into a structured graph/IR
  with explicit operation coverage.
- Add auditable FLOP, byte, and memory-movement bound generation for common SOL
  ExecBench operator families.
- Externalize AMD hardware models and validation/confidence metadata instead of
  relying only on hard-coded provisional defaults.
- Integrate improved bound artifacts with AMD-native score reports and clear
  unsupported/inexact degradation behavior.
- Add golden tests, documentation, and RDNA 4 validation coverage for the new
  modeling pipeline.

**Explicitly deferred:**
- CDNA 3 / MI300X real-hardware validation.
- CDNA 4 validation.
- NVFP4/MXFP4 hardware validation unless a suitable AMD path is separately
  established.
- Full original 124-model extraction pipeline unless it is needed only as
  reference context.

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
- Source-specific timing policy, `rocprofv3` timing evidence helpers, AMD SOL
  bound artifacts, and derived AMD-native score reports are implemented without
  mutating canonical trace JSONL or claiming NVIDIA B200/SOLAR/leaderboard
  equivalence - v1.5.
- Broader AMD SOL analyzer coverage, live `rocprofv3` collection helpers,
  dataset-runner AMD-native score reports, and v1.6 compatibility/claim
  guardrails are implemented without breaking canonical trace JSONL, public
  schemas, or primary `sol-execbench` CLI defaults - v1.6.
- Optimized scoring baseline artifacts, explicit baseline-source semantics,
  dataset integration, and baseline documentation are implemented without
  mutating canonical trace JSONL - v1.7.
- Source-specific ROCm profiler timing evidence collection, dataset integration,
  parser coverage, and auditable timing metadata are implemented - v1.7.
- Static reward-hack review now covers hidden streams, semantic caches,
  unauthorized loaders/file I/O, opaque payloads, and precision downgrade
  patterns before submitted Python source import - v1.7.
- `hipblas` has a runnable public SGEMM example and native staging tests, while
  MIOpen, CK, and rocWMMA were documented candidate categories with
  overclaiming guardrails - v1.7.
- MI300X/CDNA3 validation instructions, FP8/NVFP4 decision records, and
  evidence gates for validation claims are implemented - v1.7.
- MIOpen, Composable Kernel, and rocWMMA each have scoped native ROCm public
  examples, native staging tests, dependency diagnostics, RDNA 4 E2E
  registration, and support-status docs - v1.8.
- Former NVIDIA library/DSL compatibility paths are mapped to supported ROCm
  examples or kept explicitly as PyTorch compatibility examples - v1.8.
- v1.8 library validation claims are scoped to RDNA 4; CDNA 3 and CDNA 4
  validation remain deferred - v1.8.
- AMD SOL/SOLAR operator modeling now provides rich per-node formulas, formula
  inputs, read/write/intermediate/movement/total byte evidence, confidence,
  rationale, and legacy v1 `WorkEstimate` adaptation for GEMM/BMM, elementwise,
  activation, reduction, normalization, softmax, data movement, and dtype
  conversion families - v1.9 Phase 43.
- AMD SOL/SOLAR bound modeling now has strict AMD hardware model artifacts,
  structured bound graph IR extraction, rich operator estimates, v2 bound
  sidecars, coverage/warning semantics, AMD-native score and dataset
  integration, documentation, and public-contract guardrails - v1.9.
- v1.9 validation claims are scoped to RDNA 4 (`gfx1200`); CDNA 3 / MI300X
  real-hardware validation and CDNA 4 validation remain deferred - v1.9.
- Paper-aligned automatic SOLAR derivation now has sidecar-only fixture
  contracts, semantic provenance, formula/byte/bound evidence for scoped
  families, degraded MoE and SSM/Mamba evidence, coverage/status sidecars,
  AMD-native score guards, dataset-runner generated SOLAR sidecars,
  `derived_evidence_refs`, documentation, and public/claim guardrails - v1.10.

### Active

- Dataset acquisition/layout, paper parity inventory, ROCm compatibility
  classification, ready-subset execution closure, and parity gap reporting for
  v1.11.

### Out of Scope

- Maintaining CUDA/NVIDIA runtime compatibility - ROCm is the target platform for this port.
- Preserving NVIDIA-specific dependency implementations when a ROCm replacement is required - equivalent behavior matters more than retaining original code.
- Guaranteeing one-to-one high-performance replacements for every NVIDIA DSL - some former CUTLASS/cuDNN/CuTe/cuTile categories remain documented compatibility examples or candidates.
- Claiming CDNA 3 hardware validation before a full `gfx94*` suite pass is recorded.
- Claiming NVIDIA leaderboard equivalence or B200/SOLAR equivalence for AMD-native scores.
- Reintroducing NVIDIA/CUDA runtime parity as a goal - this project remains a
  ROCm-only port.
- Recreating the original paper's 124-model extraction and curation pipeline in
  v1.7 - explicitly deferred.
- Completing full upstream SOLAR parity in v1.7 - explicitly deferred.
- Claiming NVFP4/MXFP4 hardware validation without suitable AMD hardware
  evidence - explicitly deferred.
- CDNA 3 / MI300X real-hardware validation in v1.9 - explicitly deferred so
  the milestone can focus on modeling correctness and RDNA 4 validation.
- CDNA 4 hardware validation in v1.9 - explicitly deferred.
- Full original 124-model extraction pipeline in v1.9 - deferred unless needed
  only as reference context for bound-modeling decisions.
- Full upstream NVlabs/SOLAR equivalence in v1.11 - deferred so the milestone
  can focus on public dataset parity inventory and ROCm execution closure.
- Full 235-problem real-hardware validation in v1.11 - deferred unless the
  environment and runtime budget are proven sufficient during the milestone.

## Context

The current codebase is now ROCm-oriented. `docker/Dockerfile` uses a ROCm
development base image. Python dependency declarations point PyTorch and
torchvision at ROCm wheels. The package remains a Python CLI benchmark runner:
`src/sol_execbench/cli/main.py` loads problem JSON, `ProblemPackager` stages
files and injects AMD architecture flags, and `src/sol_execbench/driver/templates/`
contains the HIP-aware build and eval subprocess scripts.

Current high-risk areas after v1.10 are real MI300X/CDNA3 hardware validation,
CDNA 4 validation, FP8/NVFP4/MXFP4 evidence, hosted leaderboard/service
behavior, and original-paper dataset extraction. v1.10 intentionally completed
the local SOLAR derivation system itself, not dataset-scale extraction or
commercial GPU validation. No CDNA 3 / MI300X validation claim should be made
until a full adapted suite run and required environment evidence are archived in
a later milestone.

The v1.8 milestone was scoped to RDNA 4 validation only. CDNA 3 and CDNA 4
library validation remain intentionally deferred; current artifacts preserve
schema, documentation, and no-claim guardrails for those architectures.

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
| Build a SOLAR-like AMD model | User selected a SOLAR-like pipeline over a minimal configuration-only roofline model for v1.5. | Validated in v1.5 |
| Replace default timing with profiler timing | User selected replacing the default event timing path with a profiler-backed path for v1.5. | Validated in v1.5 as policy/evidence foundation |
| Accuracy beats unified timing semantics | User clarified that Triton, HIP native, and PyTorch timing semantics must be investigated separately; if one unified timing definition is inaccurate, expose operator-source-specific timer backends. | Validated in v1.5 |
| Exclude CDNA 3 hardware validation from v1.5 | User explicitly said the next milestone does not include CDNA 3 validation. | Validated in v1.5 |
| Preserve public contracts during v1.6 workflow integration | User made canonical trace JSONL, schema, and primary CLI compatibility a hard constraint for v1.6. | Validated in v1.6 |
| Keep CDNA 3 validation out of v1.6 | User said the next milestone's forced constraint still excludes CDNA 3 validation. | Validated in v1.6 |
| Expose source-specific timer chimneys when needed | User required timing accuracy over one unified口径 for Triton, HIP, and PyTorch sources. | Validated in v1.6 |
| Treat NVIDIA/B200 parity as non-scope | User clarified the project is ROCm-only and original NVIDIA hardware should not be counted as a missing feature. | Validated in v1.7 |
| Use MI300X as commercial validation target | User selected AMD MI300X/CDNA3 as the later real-hardware validation example, including FP8 validation when available. | Validated in v1.7 readiness docs |
| Defer dataset extraction and full SOLAR parity | User explicitly postponed the original paper's extraction pipeline and deeper SOLAR parity work. | Deferred |
| Prioritize baseline, timing, reward-hack, and ROCm libraries | User marked scoring baseline, evaluation timing loop, reward-hack defenses, and ROCm library migration as the implementation priorities. | Validated in v1.7 |
| Defer NVFP4/MXFP4 validation | User noted MI300X can validate FP8 later, but no hardware is available for NVFP4/MXFP4 validation now. | Deferred |
| Complete ROCm library replacement ecosystem on RDNA 4 | User requested the next milestone focus on thoroughly resolving incomplete ROCm library replacements, and clarified only RDNA 4 validation is in scope. | Validated in v1.8 |
| Focus v1.9 on AMD SOL/SOLAR bound modeling | User requested the next milestone completely solve AMD SOL/SOLAR bound modeling, with validation scoped to RDNA 4. | Validated in v1.9 |
| Defer CDNA 3 / MI300X and CDNA 4 validation from v1.9 | User clarified CDNA 3 validation and MI300X real-hardware validation should be treated as one deferred item, and CDNA 4 remains deferred. | Validated in v1.9 |
| Scope v1.10 to SOLAR derivation only | User chose to upgrade "论文级完整 SOLAR 自动推导" as a new milestone while excluding dataset extraction and real-hardware validation from the milestone boundary. | Validated in v1.10 |
| Focus v1.11 on public dataset parity and execution closure | User requested the next milestone target "论文数据集 parity + ROCm execution closure" and then narrowed it to dataset acquisition/layout, inventory, compatibility classification, small ready-subset execution, and gap reporting. | Active in v1.11 |

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
*Last updated: 2026-05-23 after v1.11 milestone start*
