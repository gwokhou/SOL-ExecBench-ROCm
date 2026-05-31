# Roadmap: SOL ExecBench ROCm Port

## Milestones

- Planned **v1.19 Research Credibility Without New Hardware** -
  Phases 83-88.

- Complete **v1.18 ROCm Version Matrix via Docker** -
  Phases 78-82 (shipped 2026-05-28). See
  `.planning/milestones/v1.18-ROADMAP.md`.

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

**Active milestone:** v1.19 Research Credibility Without New Hardware.

**Status:** Roadmap created; ready to plan Phase 83.

**Next step:** Start Phase 83 with `/gsd-plan-phase 83`.

## Phases

- [x] **Phase 83: Closure Contracts And Provenance Foundation** - Researchers get deterministic, CPU-safe execution-closure contracts and provenance checks for reusable evidence. (completed 2026-05-31)
- [ ] **Phase 84: Paper Denominator Accounting And Claim Boundaries** - Researchers can account for the public benchmark denominator and evidence gaps without claiming paper parity.
- [ ] **Phase 85: Compatibility Matrix Schema Export And Semantic Diff** - Researchers and downstream tools can export Matrix schemas and compare Matrix reports semantically.
- [ ] **Phase 86: Dataset Runner Hardening Integration** - Dataset runs classify closure outcomes and resume behavior consistently without changing default benchmark semantics.
- [ ] **Phase 87: AMD SOL/SOLAR Bound Sanity Evidence** - Researchers can inspect diagnostic AMD bound sanity over existing RDNA 4 and Docker evidence only.
- [ ] **Phase 88: Documentation, Examples, And Guardrail Tests** - Public docs, examples, and CPU-safe tests keep v1.19 evidence surfaces interpretable and claim-safe.

## Phase Details

### Phase 83: Closure Contracts And Provenance Foundation
**Goal**: Researchers have a strict, deterministic execution-closure sidecar contract that explains closure status, reason codes, totals, source refs, and evidence provenance without mutating canonical benchmark contracts.
**Depends on**: Phase 82
**Requirements**: CLOS-01, CLOS-02, CLOS-03, CLOS-04
**Success Criteria** (what must be TRUE):
  1. Researcher can validate and serialize closure records with deterministic ordering, totals, source refs, and provenance fields on CPU-only fixtures.
  2. Researcher can distinguish attempted pass, attempted failure, not attempted, filtered, skipped existing pass, missing trace, missing derived evidence, and setup/runtime blockers using stable status and reason codes.
  3. Researcher can detect manifest, readiness, ready-subset, workload identity, solution mode, and evidence-requirement mismatches before accepting existing traces as reusable.
  4. Dataset tooling can call closure helper APIs without changing canonical Trace JSONL, correctness, timing, or score semantics.
**Plans**: TBD

### Phase 84: Paper Denominator Accounting And Claim Boundaries
**Goal**: Researchers can produce a deterministic paper denominator report that accounts for ready, blocked, unsupported, deferred, attempted, filtered, skipped, and evidence-missing benchmark status without presenting the accounting as paper validation.
**Depends on**: Phase 83
**Requirements**: DENOM-01, DENOM-02, DENOM-03, DENOM-04, DENOM-05
**Success Criteria** (what must be TRUE):
  1. Researcher can generate `paper_denominator_report.v1` JSON that rolls up public benchmark denominator status by problem, workload, category, readiness, closure status, and evidence gap.
  2. Researcher can inspect stable denominator reason codes that separate ready, blocked, unsupported, deferred, evidence-missing, attempted-passed, attempted-failed, filtered, skipped, and not-attempted states.
  3. Report consumers can trace denominator conclusions back to source manifest, inventory, readiness, ready-subset, closure, AMD score, AMD SOL, and SOLAR artifacts by path/ref and checksum.
  4. Researcher can read deterministic Markdown counts, evidence gaps, deferred buckets, and next-evidence hints with paper parity, upstream SOLAR parity, leaderboard authority, native-host validation, and new-hardware validation explicitly kept false.
**Plans**: TBD

### Phase 85: Compatibility Matrix Schema Export And Semantic Diff
**Goal**: Researchers and downstream evidence producers can export strict Matrix JSON Schemas and compare ROCm Compatibility Matrix reports by semantic changes while preserving Docker/native-host and authority boundaries.
**Depends on**: Phase 84
**Requirements**: MATRIX-01, MATRIX-02, MATRIX-03, MATRIX-04, MATRIX-05
**Success Criteria** (what must be TRUE):
  1. Researcher can export JSON Schema for `MatrixEntry` and `RocmCompatibilityMatrixReport` with schema identity, version metadata, and strict extra-field behavior.
  2. Researcher can diff two Matrix reports by Target identity and validation scope, seeing added, removed, unchanged, and changed entries.
  3. Researcher can identify semantic Matrix changes across status, reason code, requested Target values, observed evidence, dependency policy, Docker image metadata, clock/evidence metadata, artifact refs, and claim boundaries.
  4. Researcher can consume both JSON diff output and a severity-ranked human summary for validation downgrade, mixed-version drift, runtime unavailability, image/dependency drift, GPU architecture drift, and claim-boundary escalation.
  5. Matrix schema and diff outputs remain diagnostic and cannot upgrade Docker container evidence into native-host validation, score authority, paper-parity authority, or leaderboard authority.
**Plans**: TBD

### Phase 86: Dataset Runner Hardening Integration
**Goal**: Dataset runner executions reuse closure helpers to classify resume, reuse, skipped, failed, missing-evidence, and unattempted outcomes deterministically while preserving existing default execution behavior.
**Depends on**: Phase 83
**Requirements**: RUNNER-01, RUNNER-02, RUNNER-03, RUNNER-04, RUNNER-05
**Success Criteria** (what must be TRUE):
  1. Researcher can resume or reuse dataset output only when ready-subset, readiness, manifest, problem, workload, solution, and evidence provenance match the selected run configuration.
  2. Researcher can see existing passing traces marked `skipped_existing_pass` only when provenance matches, while `--rerun` records a fresh attempt.
  3. Researcher can inspect closure output that classifies build failures, runtime failures, timeouts, nonzero CLI exits, correctness failures, missing traces, missing derived evidence, and skipped/unattempted workloads with stable reason codes and bounded log refs.
  4. Existing dataset runner defaults behave as before unless a mismatch, missing evidence, or explicit closure option requires a diagnostic stop or sidecar status.
  5. Closure writes are deterministic and avoid credentials, proprietary kernels, raw dataset payloads, unnecessary absolute paths, and unbounded logs.
**Plans**: TBD

### Phase 87: AMD SOL/SOLAR Bound Sanity Evidence
**Goal**: Researchers can generate diagnostic AMD SOL/SOLAR bound sanity reports over existing RDNA 4 and Docker evidence while keeping score eligibility and hardware-validation claims unchanged.
**Depends on**: Phase 84
**Requirements**: SANITY-01, SANITY-02, SANITY-03, SANITY-04
**Success Criteria** (what must be TRUE):
  1. Researcher can generate `amd_bound_sanity.v1` over existing trace, closure, AMD SOL, SOLAR derivation, AMD score, and compatibility evidence refs/checksums.
  2. Researcher can inspect artifact availability, aggregate statuses, coverage summaries, warnings, and evidence gaps for AMD SOL/SOLAR evidence.
  3. Researcher can distinguish scored, degraded, unscored, unsupported, provisional, and missing-evidence states without changing AMD-native score semantics or score eligibility rules.
  4. Bound sanity output surfaces provisional RDNA 4 model risk while explicitly avoiding upstream SOLAR equivalence, model-validation, paper-parity, leaderboard, CDNA 3, MI300X, CDNA 4, and native-host validation claims.
  5. Bound sanity checks run without new hardware probes, Docker privilege changes, or dependency relocking.
**Plans**: TBD

### Phase 88: Documentation, Examples, And Guardrail Tests
**Goal**: Researchers can understand, reproduce, and safely interpret v1.19 evidence surfaces through documentation, representative examples, and CPU-safe tests that preserve public contracts and claim boundaries.
**Depends on**: Phase 87
**Requirements**: DOCS-01, DOCS-02, DOCS-03, DOCS-04, DOCS-05
**Success Criteria** (what must be TRUE):
  1. Researcher can follow docs to generate and interpret denominator reports, closure hardening outputs, Matrix schema exports, Matrix diffs, and AMD bound sanity reports.
  2. Documentation states that v1.19 does not add full 235-problem paper validation, upstream SOLAR parity, score authority, leaderboard readiness, CDNA 3/MI300X/CDNA4 validation, or native-host ROCm Matrix validation.
  3. CPU-safe tests cover denominator accounting, closure serialization/provenance, Matrix schema export, Matrix diff semantics, dataset-runner closure classification, AMD bound sanity reports, and docs wording guardrails.
  4. Public examples or fixture reports show representative JSON/Markdown artifact shapes with bounded logs, relative refs, checksums, and explicit authority-false or diagnostic-only interpretation.
  5. Existing public contracts remain stable: canonical Trace, Definition, Workload, Solution, correctness, timing, score, and evaluator contract semantics are unchanged by v1.19 reporting features.
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 83 -> 84 -> 85 -> 86 -> 87 -> 88.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 83. Closure Contracts And Provenance Foundation | 2/2 | Complete   | 2026-05-31 |
| 84. Paper Denominator Accounting And Claim Boundaries | 0/TBD | Not started | - |
| 85. Compatibility Matrix Schema Export And Semantic Diff | 0/TBD | Not started | - |
| 86. Dataset Runner Hardening Integration | 0/TBD | Not started | - |
| 87. AMD SOL/SOLAR Bound Sanity Evidence | 0/TBD | Not started | - |
| 88. Documentation, Examples, And Guardrail Tests | 0/TBD | Not started | - |
