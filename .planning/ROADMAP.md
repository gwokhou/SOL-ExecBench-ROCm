# Roadmap: SOL ExecBench ROCm Port

## Milestones

- Active **v1.15 Research-Grade ROCm Benchmark Release** —
  Phases 64-67. This milestone turns the ROCm port into a small, complete,
  researcher-friendly release with claim boundaries, a curated benchmark slice,
  cookbooks, and reproducibility closure.

- Complete **v1.14 Optional rocprofv3 Profiling Evidence** —
  Phases 61-63 (shipped 2026-05-25). See
  `.planning/milestones/v1.14-ROADMAP.md`.

- Complete **v1.13 ROCm Runtime Evidence and Environment Diagnostics** —
  Phases 58-60 (shipped 2026-05-25). See
  `.planning/milestones/v1.13-ROADMAP.md`.

- Complete **v1.12 Evaluator Contract Metadata and Boundary Guardrails** —
  retroactive quick-task milestone (shipped 2026-05-25). See
  `.planning/milestones/v1.12-ROADMAP.md`.

- Complete **v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure** —
  Phases 53-57 (shipped 2026-05-23). See
  `.planning/milestones/v1.11-ROADMAP.md`.

- Complete **v1.10 Paper-Aligned SOLAR Automatic Derivation** —
  Phases 47-52 (shipped 2026-05-23). See
  `.planning/milestones/v1.10-ROADMAP.md`.

- Complete **v1.9 AMD SOL/SOLAR Bound Modeling Completion** —
  Phases 41-46 (shipped 2026-05-23). See
  `.planning/milestones/v1.9-ROADMAP.md`.

- Complete **v1.8 ROCm Library Ecosystem Completion** —
  Phases 36-40 (shipped 2026-05-22). See
  `.planning/milestones/v1.8-ROADMAP.md`.

- Complete **v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration** —
  Phases 31-35 (shipped 2026-05-22). See
  `.planning/milestones/v1.7-ROADMAP.md`.

- Complete **v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow** —
  Phases 27-30 (shipped 2026-05-22). See
  `.planning/milestones/v1.6-ROADMAP.md`.

- Complete **v1.5 AMD-native SOL Scoring and ROCm Profiler Timing** —
  Phases 23-26 (shipped 2026-05-22). See
  `.planning/milestones/v1.5-ROADMAP.md`.

- Complete **v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness** —
  shipped 2026-05-22. See `.planning/milestones/v1.4-ROADMAP.md`.

- Complete **v1.3 Non-CDNA Issue Closure** — shipped 2026-05-22. See
  `.planning/milestones/v1.3-ROADMAP.md`.

- Complete **v1.2 Engineering Practice Harvest and Compatibility Guardrails** —
  shipped 2026-05-22. See `.planning/milestones/v1.2-ROADMAP.md`.

- Complete **v1.1 CDNA 3 Support and Migration Closure** — shipped
  2026-05-21. See `.planning/milestones/v1.1-ROADMAP.md`.

- Complete **v1.0 ROCm Port** — shipped 2026-05-21. See
  `.planning/milestones/v1.0-ROADMAP.md`.

## Current Position

**Active milestone:** v1.15 Research-Grade ROCm Benchmark Release.

**Status:** defining requirements and roadmap.

**Milestone goal:** make the ROCm port usable as a small, complete,
research-grade benchmark preview before attempting full paper parity.

## Active Phase Roadmap

### Phase 64: Claim Boundary and Researcher Positioning

**Status:** Planned
**Milestone:** v1.15 Research-Grade ROCm Benchmark Release
**Goal:** Make the project's current claims, non-claims, evidence requirements,
and research positioning explicit and test-protected.

**Scope:**
- Add a claim-boundary guide such as `docs/CLAIMS.md`.
- Map allowed claims to required evidence artifacts and commands.
- Document unsupported claims and upgrade paths.
- Add or update guardrail tests that protect public wording.

**Requirement IDs:** CLAIM-01, CLAIM-02, CLAIM-03, CLAIM-04

**Success criteria:**
1. A reader can tell what results are valid ROCm-port evidence today.
2. Unsupported paper/leaderboard/hardware parity claims are explicit.
3. Claim upgrades require concrete evidence rather than wording changes.
4. Tests protect the most important claim boundaries.

### Phase 65: Curated ROCm Benchmark Slice

**Status:** Planned
**Milestone:** v1.15 Research-Grade ROCm Benchmark Release
**Goal:** Define and exercise a representative, bounded ROCm benchmark slice
that can be reproduced without implying full paper parity.

**Scope:**
- Select 10-20 representative problems or a smaller stable slice if repository
  fixtures make that more reliable.
- Cover PyTorch ROCm, Triton ROCm, HIP/C++, and at least one ROCm native
  library path when available.
- Record selection criteria, workload scope, hardware assumptions, and
  exclusions.
- Ensure execution flows through existing CLI or dataset-runner paths.
- Record expected traces, sidecars, score artifacts, or unscored reasons.

**Requirement IDs:** SLICE-01, SLICE-02, SLICE-03, SLICE-04

**Success criteria:**
1. The curated slice is deterministic and documented.
2. Slice execution uses existing benchmark paths.
3. Artifact expectations are explicit for each selected problem.
4. Missing hardware or score evidence produces explicit unavailable/unscored
   states instead of silent omission.

### Phase 66: Researcher Workflows and Cookbooks

**Status:** Planned
**Milestone:** v1.15 Research-Grade ROCm Benchmark Release
**Goal:** Give GPU kernel researchers and deep developers a direct path from
first run to extending kernels, interpreting artifacts, and running agent or
compiler experiments.

**Scope:**
- Add `docs/RESEARCHER-GUIDE.md`.
- Add cookbook recipes for single-kernel evaluation, HIP/Triton solution
  adaptation, curated-slice execution, AMD-native score evidence, and
  `rocprofv3` diagnostics.
- Explain artifact interpretation for traces, environment sidecars, profile
  sidecars, AMD-native score reports, and readiness/closure reports.
- Describe agent optimizer and compiler/backend experiment boundaries.

**Requirement IDs:** RESEARCH-01, RESEARCH-02, RESEARCH-03, RESEARCH-04, COOK-01, COOK-02

**Success criteria:**
1. Kernel authors can run and modify one solution from docs alone.
2. Compiler/backend researchers can identify integration points and schemas.
3. Agent researchers can use the harness without bypassing guardrails.
4. Cookbook commands are copy-pasteable or clearly state prerequisites.

### Phase 67: Release Closure and Reproducibility Bundle

**Status:** Planned
**Milestone:** v1.15 Research-Grade ROCm Benchmark Release
**Goal:** Package the curated research preview into a reproducibility closure
that records commands, artifacts, validation status, and remaining gaps.

**Scope:**
- Add a v1.15 release closure document.
- Record curated-slice commands and expected artifact families.
- Summarize pass/fail/skip/unavailable/unscored semantics.
- Link the closure back to claim boundaries and future paper-parity work.

**Requirement IDs:** REPRO-01, REPRO-02

**Success criteria:**
1. The release closure can be used as a checklist for reproducing v1.15.
2. The closure distinguishes local evidence from paper parity and leaderboard
   readiness.
3. Known gaps are explicit and mapped to future requirements.
4. The next likely milestone is clear.

## Progress

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v1.15 Research-Grade ROCm Benchmark Release | 64-67 | 0/4 | Active | - |
| v1.14 Optional rocprofv3 Profiling Evidence | 61-63 | 3/3 | Complete | 2026-05-25 |
| v1.13 ROCm Runtime Evidence and Environment Diagnostics | 58-60 | 5/5 | Complete | 2026-05-25 |
| v1.12 Evaluator Contract Metadata and Boundary Guardrails | none | quick task 260524-xb3 | Complete | 2026-05-25 |
| v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure | 53-57 | 14/14 | Complete | 2026-05-23 |
| v1.10 Paper-Aligned SOLAR Automatic Derivation | 47-52 | 23/23 | Complete | 2026-05-23 |
| v1.9 AMD SOL/SOLAR Bound Modeling Completion | 41-46 | 17/17 | Complete | 2026-05-23 |
| v1.8 ROCm Library Ecosystem Completion | 36-40 | 5/5 | Complete | 2026-05-22 |
| v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration | 31-35 | 5/5 | Complete | 2026-05-22 |
| v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow | 27-30 | 4/4 | Complete | 2026-05-22 |
| v1.5 AMD-native SOL Scoring and ROCm Profiler Timing | 23-26 | 4/4 | Complete | 2026-05-22 |
| v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness | - | - | Complete | 2026-05-22 |
| v1.3 Non-CDNA Issue Closure | - | - | Complete | 2026-05-22 |
| v1.2 Engineering Practice Harvest and Compatibility Guardrails | - | - | Complete | 2026-05-22 |
| v1.1 CDNA 3 Support and Migration Closure | - | - | Complete | 2026-05-21 |
| v1.0 ROCm Port | - | - | Complete | 2026-05-21 |

## Future Candidate Work

- Static kernel evidence with RGA/code-object analysis and GPUOpen ISA
  classification.
- Original paper-scale 124-model / 235-problem extraction and curation.
- MI300X, CDNA 3, and CDNA 4 real-hardware validation.
- NVFP4 and MXFP4 validation if a suitable AMD hardware path exists.
- Hosted leaderboard or submission service.
- NVIDIA Blackwell/B200 comparison methodology, if ever scoped as a separate
  non-ROCm claim analysis effort.
