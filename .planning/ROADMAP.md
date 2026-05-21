# Roadmap: SOL ExecBench ROCm Port

**Created:** 2026-05-21
**Mode:** standard
**Granularity:** standard

## Overview

This roadmap ports SOL ExecBench to a ROCm-only implementation through
horizontal technical layers. The order follows hard dependencies: establish the
ROCm environment, define schema/build targets, port evaluation and timing, then
migrate examples, validate hardware, and finish documentation/compliance.

## Phase Status

- [x] **Phase 1: ROCm Environment Baseline** (completed 2026-05-21)
- [x] **Phase 2: ROCm Schema and Native Build Layer** (completed 2026-05-21)
- [x] **Phase 3: ROCm Evaluation, Timing, and Hardware Introspection** (completed 2026-05-21)
- [ ] **Phase 4: ROCm Library and Example Migration**
- [ ] **Phase 5: ROCm Test Suite and Hardware Validation**
- [ ] **Phase 6: Documentation, Analysis Workflow, and Compliance**

## Phases

### Phase 1: ROCm Environment Baseline
**Goal:** Replace the CUDA/NVIDIA development environment with a reproducible ROCm >= 7.0 baseline.

**Requirements:** ENV-01, ENV-02, ENV-03, ENV-04, SCFG-03

**Plans:** 4 plans

Plans:
- [ ] 01-01-PLAN.md - Replace Dockerfile, Docker run flags, and entrypoint with ROCm baseline behavior.
- [ ] 01-02-PLAN.md - Replace CUDA/NVIDIA Python dependency sources with ROCm wheel sources and refreshed lock data.
- [ ] 01-03-PLAN.md - Add ROCm dependency smoke tests for ROCm runtime, HIP, PyTorch, Triton, and selected ROCm libraries.
- [ ] 01-04-PLAN.md - Remove superseded CUDA/NVIDIA Docker dependency smoke tests after ROCm replacements exist.

**Success Criteria:**
1. Docker image builds from a ROCm base and no longer depends on `nvidia/cuda`.
2. Container verifies ROCm runtime, HIP compiler, PyTorch ROCm, Triton ROCm, and selected ROCm libraries.
3. CUDA wheel indexes and NVIDIA-only Python dependencies are removed or replaced.
4. Docker dependency tests distinguish missing ROCm tooling from test failures.

### Phase 2: ROCm Schema and Native Build Layer
**Goal:** Make solution metadata and native compilation express ROCm targets and build HIP/C++ solutions.

**Requirements:** SCFG-01, SCFG-02, BUILD-01, BUILD-02, BUILD-03, BUILD-04

**Success Criteria:**
1. Solution schemas represent HIP/C++ and AMD hardware targets without breaking existing problem file compatibility unnecessarily.
2. `ProblemPackager` stages HIP/C++ sources and applies AMD gfx target handling instead of CUDA `-gencode`.
3. Native build template compiles a minimal HIP/C++ solution into the existing shared-object loading contract.
4. Compile failures produce actionable logs while preserving trace JSON discipline.
5. HIPIFY or an equivalent audit command reports remaining CUDA-specific source patterns.

### Phase 3: ROCm Evaluation, Timing, and Hardware Introspection
**Goal:** Port the isolated evaluation runtime and benchmark-integrity mechanisms to ROCm.

**Requirements:** EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, PROF-01, PROF-02, PROF-03, PROF-04, PROF-05

**Success Criteria:**
1. PyTorch ROCm, Triton ROCm, and HIP/C++ solutions produce valid trace JSONL through `eval_driver.py`.
2. Destination-passing and return-value conventions continue to work.
3. Input generation, output allocation/normalization, and correctness checks preserve existing semantics.
4. Timing no longer depends on CUPTI/CUDA-only APIs and passes asynchronous-work hiding tests adapted for ROCm.
5. Environment and clock/hardware reports use AMD/ROCm tooling and include AMD GPU plus ROCm versions.

### Phase 4: ROCm Library and Example Migration
**Goal:** Migrate public examples and original solution categories to ROCm-compatible implementations or documented alternatives.

**Requirements:** LIB-01, LIB-02, LIB-03, LIB-04, LIB-05, LIB-06, LIB-07

**Success Criteria:**
1. PyTorch, Triton, and HIP/C++ examples run under ROCm and pass adapted example tests.
2. CUTLASS-style examples are replaced or reimplemented using ROCm-appropriate libraries or kernels where feasible.
3. cuDNN-style examples are replaced or reimplemented using MIOpen or ROCm-appropriate kernels where feasible.
4. CuTe DSL and cuTile-style examples have ROCm replacements, reimplementations, or explicit feasibility notes.
5. Replacement choices across rocBLAS/hipBLASLt, MIOpen, Composable Kernel, rocWMMA, hipCUB, rocPRIM, and rocThrust are documented.

### Phase 5: ROCm Test Suite and Hardware Validation
**Goal:** Convert the test suite to ROCm semantics and prove it passes on RDNA 4 and CDNA 3.

**Requirements:** TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-06

**Success Criteria:**
1. Unit, driver, benchmark-helper, Docker dependency, example, and e2e tests run against ROCm assumptions.
2. Hardware markers distinguish unavailable ROCm, unsupported AMD architecture, RDNA 4, and CDNA 3.
3. Reward-hack tests remain active and pass under ROCm.
4. Full adapted suite passes in a ROCm >= 7.0 environment on RDNA 4.
5. Full adapted suite passes in a ROCm >= 7.0 environment on CDNA 3.

### Phase 6: Documentation, Analysis Workflow, and Compliance
**Goal:** Make the ROCm port usable and legally clean for researchers and developers.

**Requirements:** SCFG-04, DOC-01, DOC-02, DOC-03, DOC-04, DOC-05

**Success Criteria:**
1. README and setup docs describe ROCm installation, Docker usage, dataset setup, and local evaluation.
2. Schema docs describe ROCm-supported languages, hardware targets, and known limitations.
3. Profiling/analyze docs explain the ROCm-native tooling path.
4. License and third-party notices reflect retained and replacement dependencies.
5. Unsupported NVIDIA-equivalent features and known gaps are documented clearly.

## Requirement Coverage

| Requirement | Phase |
|-------------|-------|
| ENV-01 | Phase 1 |
| ENV-02 | Phase 1 |
| ENV-03 | Phase 1 |
| ENV-04 | Phase 1 |
| SCFG-01 | Phase 2 |
| SCFG-02 | Phase 2 |
| SCFG-03 | Phase 1 |
| SCFG-04 | Phase 6 |
| BUILD-01 | Phase 2 |
| BUILD-02 | Phase 2 |
| BUILD-03 | Phase 2 |
| BUILD-04 | Phase 2 |
| EVAL-01 | Phase 3 |
| EVAL-02 | Phase 3 |
| EVAL-03 | Phase 3 |
| EVAL-04 | Phase 3 |
| EVAL-05 | Phase 3 |
| PROF-01 | Phase 3 |
| PROF-02 | Phase 3 |
| PROF-03 | Phase 3 |
| PROF-04 | Phase 3 |
| PROF-05 | Phase 3 |
| LIB-01 | Phase 4 |
| LIB-02 | Phase 4 |
| LIB-03 | Phase 4 |
| LIB-04 | Phase 4 |
| LIB-05 | Phase 4 |
| LIB-06 | Phase 4 |
| LIB-07 | Phase 4 |
| TEST-01 | Phase 5 |
| TEST-02 | Phase 5 |
| TEST-03 | Phase 5 |
| TEST-04 | Phase 5 |
| TEST-05 | Phase 5 |
| TEST-06 | Phase 5 |
| DOC-01 | Phase 6 |
| DOC-02 | Phase 6 |
| DOC-03 | Phase 6 |
| DOC-04 | Phase 6 |
| DOC-05 | Phase 6 |

**Coverage:** 39/39 v1 requirements mapped.

---
*Roadmap created: 2026-05-21*
