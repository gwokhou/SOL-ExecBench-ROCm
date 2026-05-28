# Roadmap: SOL ExecBench ROCm Port

## Milestones

- **v1.18 ROCm Version Matrix via Docker** - Phases 78-82 (in progress)

- Complete **v1.17 Static Kernel Evidence** -
  Phases 73-77 (shipped 2026-05-26). See
  `.planning/milestones/v1.17-ROADMAP.md`.

- Complete **v1.16 ROCm Toolchain Research and Capability Routing** -
  Phases 68-72 (shipped 2026-05-25). See
  `.planning/milestones/v1.16-ROADMAP.md`.

- Complete **v1.15 Research-Grade ROCm Benchmark Release** -
  Phases 64-67 (shipped 2026-05-25). See
  `.planning/milestones/v1.15-ROADMAP.md`.

- Complete **v1.14 Optional rocprofv3 Profiling Evidence** -
  Phases 61-63 (shipped 2026-05-25). See
  `.planning/milestones/v1.14-ROADMAP.md`.

- Complete **v1.13 ROCm Runtime Evidence and Environment Diagnostics** -
  Phases 58-60 (shipped 2026-05-25). See
  `.planning/milestones/v1.13-ROADMAP.md`.

- Complete **v1.12 Evaluator Contract Metadata and Boundary Guardrails** -
  retroactive quick-task milestone (shipped 2026-05-25). See
  `.planning/milestones/v1.12-ROADMAP.md`.

- Complete **v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure** -
  Phases 53-57 (shipped 2026-05-23). See
  `.planning/milestones/v1.11-ROADMAP.md`.

- Complete **v1.10 Paper-Aligned SOLAR Automatic Derivation** -
  Phases 47-52 (shipped 2026-05-23). See
  `.planning/milestones/v1.10-ROADMAP.md`.

- Complete **v1.9 AMD SOL/SOLAR Bound Modeling Completion** -
  Phases 41-46 (shipped 2026-05-23). See
  `.planning/milestones/v1.9-ROADMAP.md`.

- Complete **v1.8 ROCm Library Ecosystem Completion** -
  Phases 36-40 (shipped 2026-05-22). See
  `.planning/milestones/v1.8-ROADMAP.md`.

- Complete **v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration** -
  Phases 31-35 (shipped 2026-05-22). See
  `.planning/milestones/v1.7-ROADMAP.md`.

- Complete **v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow** -
  Phases 27-30 (shipped 2026-05-22). See
  `.planning/milestones/v1.6-ROADMAP.md`.

- Complete **v1.5 AMD-native SOL Scoring and ROCm Profiler Timing** -
  Phases 23-26 (shipped 2026-05-22). See
  `.planning/milestones/v1.5-ROADMAP.md`.

- Complete **v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness** -
  shipped 2026-05-22. See `.planning/milestones/v1.4-ROADMAP.md`.

- Complete **v1.3 Non-CDNA Issue Closure** - shipped 2026-05-22. See
  `.planning/milestones/v1.3-ROADMAP.md`.

- Complete **v1.2 Engineering Practice Harvest and Compatibility Guardrails** -
  shipped 2026-05-22. See `.planning/milestones/v1.2-ROADMAP.md`.

- Complete **v1.1 CDNA 3 Support and Migration Closure** - shipped
  2026-05-21. See `.planning/milestones/v1.1-ROADMAP.md`.

- Complete **v1.0 ROCm Port** - shipped 2026-05-21. See
  `.planning/milestones/v1.0-ROADMAP.md`.

## Current Position

**Active milestone:** v1.18 ROCm Version Matrix via Docker.

**Status:** Phase 78 complete and ready for verification. Next step: Phase 79 planning.

## Phases

- [x] **Phase 78: Matrix Contract And Claim Guardrails** - Define diagnostic Matrix Entry semantics, bounded statuses, and claim boundaries. (completed 2026-05-28)
- [ ] **Phase 79: Docker Matrix Selection And Preflight** - Let users select declared ROCm Docker targets with deterministic image and device preflight behavior.
- [ ] **Phase 80: uv And PyTorch ROCm Wheel Coordination** - Record and enforce per-target dependency policy so wrong or unavailable PyTorch ROCm stacks are classified before validation.
- [ ] **Phase 81: Runtime Evidence And Compatibility Reports** - Collect scoped host/container/Python/toolchain/GPU evidence and emit per-target plus aggregate compatibility reports.
- [ ] **Phase 82: Validation Workflow, Docs, And CI Guardrails** - Document claim boundaries and add CPU-safe, Docker-script, and marker-gated validation coverage.

## Phase Details

### Phase 78: Matrix Contract And Claim Guardrails

**Goal**: Users and downstream tools can interpret ROCm compatibility Matrix Entries with stable diagnostic semantics and explicit claim boundaries.
**Depends on**: Phase 77
**Requirements**: MATRIX-01, MATRIX-02, MATRIX-03, MATRIX-04, MATRIX-05, MATRIX-06
**Success Criteria** (what must be TRUE):

  1. User can read a `sol_execbench.rocm_compatibility_matrix.v1` report and identify each Matrix Entry's Target, observed evidence, status, reason codes, artifacts, and claim boundaries.
  2. User can distinguish requested Target values from observed host, container, Python dependency, toolchain, and GPU evidence in every Matrix Entry.
  3. User can rely on the bounded status vocabulary: `host_validated`, `container_validated`, `mixed_version`, `pytorch_wheel_unavailable`, `runtime_unavailable`, and `not_tested`.
  4. User can verify that compatibility evidence is diagnostic-only and never grants score, paper-parity, or leaderboard authority.
  5. User never sees Docker container validation described as native host validation; `host_validated` requires direct native-host evidence.

**Plans**: 2 plans
Plans:
**Wave 1**

- [x] 78-01-PLAN.md — Define strict Matrix Entry contract, Target/observed evidence schema, bounded statuses, artifacts, and diagnostic claim flags.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 78-02-PLAN.md — Enforce mixed-version blocking, debug override limits, host/container claim separation, and sidecar-only public guardrails.

### Phase 79: Docker Matrix Selection And Preflight

**Goal**: Users can select declared ROCm Docker Targets and get conservative preflight classification before benchmark execution.
**Depends on**: Phase 78
**Requirements**: DOCKER-01, DOCKER-02, DOCKER-03, DOCKER-04, DOCKER-05
**Success Criteria** (what must be TRUE):

  1. User can select checked-in ROCm Docker Targets for configured 7.0.x, 7.1.x, and 7.2.x entries while the existing ROCm 7.1 default path still works.
  2. User can see exact requested image repository/tag, resolved digest when available, and Docker build arguments in the compatibility evidence.
  3. User is blocked from unknown ROCm Targets unless they explicitly choose an unsafe or untested override.
  4. User receives `runtime_unavailable` before benchmark execution when Docker context, `/dev/kfd`, `/dev/dri`, or GPU device access is unavailable.

**Plans**: TBD

### Phase 80: uv And PyTorch ROCm Wheel Coordination

**Goal**: Users can tell whether a selected ROCm Target has a matching PyTorch ROCm dependency stack before any clean validation claim is made.
**Depends on**: Phase 79
**Requirements**: DEPS-01, DEPS-02, DEPS-03, DEPS-04, DEPS-05, DEPS-06, DEPS-07
**Success Criteria** (what must be TRUE):

  1. User can inspect each Target's PyTorch ROCm wheel/index policy, including expected local-version tag and uv index or lock strategy.
  2. User can keep using the default ROCm 7.1 dependency path unless a per-Target dependency workflow is explicitly selected and recorded.
  3. User sees missing or unsupported PyTorch ROCm wheels classified as `pytorch_wheel_unavailable`, not benchmark failures.
  4. User sees CPU, CUDA, wrong-index, wrong-ROCm, Triton, or toolchain mismatches classified as `mixed_version`.
  5. User is blocked from illegal mixed-version validation by default, while an explicit debug override can continue probes or smoke execution without eligibility for clean validation or authority claims.

**Plans**: TBD

### Phase 81: Runtime Evidence And Compatibility Reports

**Goal**: Users can collect scoped runtime evidence and review per-Target and aggregate compatibility reports without changing benchmark semantics.
**Depends on**: Phase 80
**Requirements**: EVID-01, EVID-02, EVID-03, EVID-04, EVID-05, EVID-06
**Success Criteria** (what must be TRUE):

  1. User can inspect host ROCm/driver/device-node evidence separately from container ROCm user-space and toolchain evidence.
  2. User can inspect Python and dependency evidence including `torch.__version__`, `torch.version.hip`, `torch.version.cuda`, PyTorch device availability, and `triton-rocm` status.
  3. User can inspect GPU metadata including device count, device name, detected `gfx*` architecture, and visible-device environment variables when available.
  4. User can find per-Target compatibility JSON and an aggregate compatibility matrix JSON with status counts.
  5. User can distinguish setup failures, dependency failures, and benchmark correctness/performance results while canonical trace JSONL, scoring, timing, defaults, and exit semantics remain unchanged.

**Plans**: TBD

### Phase 82: Validation Workflow, Docs, And CI Guardrails

**Goal**: Users can follow documented validation workflows and trust automated guardrails that prevent ROCm matrix evidence from being overstated.
**Depends on**: Phase 81
**Requirements**: DOCS-01, DOCS-02, DOCS-03, DOCS-04, DOCS-05, DOCS-06
**Success Criteria** (what must be TRUE):

  1. User can read docs that explain Docker Matrix Entries validate container ROCm user-space on recorded host driver/devices and do not prove native host ROCm validation.
  2. User can read docs that explain Target/requested values versus observed host/container/Python/GPU evidence and why Target identity is required.
  3. User can read docs that illegal mixed-version Targets are blocked by default and debug override runs cannot make clean validation, score, paper-parity, or leaderboard claims.
  4. User can run CPU-safe tests for status classification, reason-code classification, schema serialization, mixed-version blocking, claim flags, docs wording, Docker Target selection, default preservation, unknown Target rejection, and command construction.
  5. User can follow marker-gated live ROCm validation guidance that records the current host ROCm 7.1.x environment as observed evidence without requiring host reinstall for ROCm 7.0.x or 7.2.x.

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 78 -> 79 -> 80 -> 81 -> 82

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 78. Matrix Contract And Claim Guardrails | v1.18 | 2/2 | Complete   | 2026-05-28 |
| 79. Docker Matrix Selection And Preflight | v1.18 | 0/TBD | Not started | - |
| 80. uv And PyTorch ROCm Wheel Coordination | v1.18 | 0/TBD | Not started | - |
| 81. Runtime Evidence And Compatibility Reports | v1.18 | 0/TBD | Not started | - |
| 82. Validation Workflow, Docs, And CI Guardrails | v1.18 | 0/TBD | Not started | - |

## Future Candidate Work

- CDNA 3 and CDNA 4 live validation with archived hardware-specific runs.
- Triton ROCm cache capture once cache-to-solution provenance is proven.
- RGA-derived VGPR, SGPR, LDS, scratch, occupancy-like, or resource-summary
  parsing after live fixture validation.

- Static instruction-family classification, static sidecar diffs, and
  static/profile kernel-name correlation.

- Dataset-level static evidence aggregation without requiring full paper-scale
  coverage.

- Standalone static artifact analysis for existing `.hsaco`, code object,
  shared object, or ELF inputs.

- Native host ROCm 7.0.x/7.1.x/7.2.x compatibility validation across supported
  GPU generations.

- Original paper-scale 124-model / 235-problem extraction and curation.
- MI300X, CDNA 3, and CDNA 4 real-hardware validation.
- Hosted leaderboard or submission service.
