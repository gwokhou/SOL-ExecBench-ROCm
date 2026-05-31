# Roadmap: SOL ExecBench ROCm Port

## Milestones

- Planned **v1.20 Cross-Report Consistency and Evaluation Stability** -
  Phases 89-93.

- Complete **v1.19 Research Credibility Without New Hardware** -
  Phases 83-88 (shipped 2026-05-31). See
  `.planning/milestones/v1.19-ROADMAP.md`.

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

**Active milestone:** v1.20 Cross-Report Consistency and Evaluation Stability.

**Status:** Roadmap created; ready to plan Phase 89.

**Next step:** Start Phase 89 with `/gsd-plan-phase 89`.

## Phases

- [ ] **Phase 89: Cross-Report Consistency Contract And Lint** - Researchers can detect contradictions across current evidence sidecars without changing benchmark contracts.
- [ ] **Phase 90: Evaluation Stability Evidence** - Researchers can inspect timing quality, variance, clock policy, and backend risk through a sidecar-only stability report.
- [ ] **Phase 91: Claim Upgrade Rules And Authority Gates** - Researchers can evaluate whether evidence satisfies explicit prerequisites for stronger validation and authority claims.
- [ ] **Phase 92: Trust Summary Integration** - Researchers can generate a concise JSON/Markdown trust summary that combines consistency, stability, closure, denominator, Matrix, score, and bound status.
- [ ] **Phase 93: Documentation, Examples, And Guardrail Tests** - Public docs, fixtures, and CPU-safe/ROCm guardrails make v1.20 artifacts interpretable and claim-safe.

## Phase Details

### Phase 89: Cross-Report Consistency Contract And Lint
**Goal**: Researchers can run a deterministic, CPU-safe consistency lint over closure, denominator, Matrix, runtime/static evidence, AMD score, AMD SOL/SOLAR, and AMD bound sanity reports.
**Depends on**: Phase 88
**Requirements**: CONS-01, CONS-02, CONS-03, CONS-04, CONS-05
**Success Criteria** (what must be TRUE):
  1. Researcher can load supported v1.19-era evidence refs through strict sidecar/report models without mutating canonical Trace, score, timing, or public schemas.
  2. Researcher can see contradictions classified for attempted/blocked drift, runtime-unavailable attempted evidence, missing-derived-evidence scored reports, and stale refs/checksums.
  3. Researcher can consume stable severity and reason codes for blockers, warnings, informational notes, and claim-boundary violations.
  4. Researcher can write deterministic JSON and Markdown summaries with bounded relative refs, checksums, and no embedded raw logs, credentials, proprietary kernels, or absolute temp paths.
  5. Consistency lint remains diagnostic-only and cannot upgrade any evidence authority.
**Plans**: TBD

### Phase 90: Evaluation Stability Evidence
**Goal**: Researchers can produce and interpret `evaluation_stability.v1` diagnostics that describe timing quality without changing canonical timing, correctness, scoring, or evaluator semantics.
**Depends on**: Phase 89
**Requirements**: STAB-01, STAB-02, STAB-03, STAB-04, STAB-05
**Success Criteria** (what must be TRUE):
  1. Researcher can validate and serialize strict stability sidecars with timing backend, warmup, repeats, runtime distribution, selected statistic, clock policy, synchronization policy, and source trace refs.
  2. Researcher can distinguish stable, noisy, insufficient-samples, missing-timing, clock-unlocked, profiler-overhead-risk, and backend-unsupported states with stable reason codes.
  3. Stability summaries compute deterministic dispersion metrics from existing timing evidence without changing trace JSONL or score behavior.
  4. A focused ROCm E2E path demonstrates real HIP/C++ or PyTorch ROCm timing evidence can emit or validate stability diagnostics.
  5. Documentation and tests preserve the boundary that stability supports interpretation but does not create correctness, score, paper-parity, native-host, or leaderboard authority.
**Plans**: TBD

### Phase 91: Claim Upgrade Rules And Authority Gates
**Goal**: Researchers can evaluate machine-readable prerequisites for diagnostic-only, container-validated, native-host-validated, score-authoritative, paper-parity-candidate, and leaderboard-ready claims.
**Depends on**: Phase 90
**Requirements**: CLAIM-01, CLAIM-02, CLAIM-03, CLAIM-04
**Success Criteria** (what must be TRUE):
  1. Researcher can inspect a versioned claim-upgrade rule set with explicit required evidence for each claim level.
  2. Claim evaluation rejects upgrades when closure, denominator, Matrix, runtime, stability, AMD score, AMD SOL/SOLAR, or hardware validation evidence is missing or contradictory.
  3. Claim evaluation outputs unmet prerequisites and next-evidence hints without silently changing report authority fields.
  4. Existing v1.19 and earlier diagnostic artifacts remain authority-false unless every required prerequisite is proven.
**Plans**: TBD

### Phase 92: Trust Summary Integration
**Goal**: Researchers can generate a concise trust summary that combines consistency, stability, claim-upgrade, closure, denominator, Matrix, AMD score, and AMD SOL/SOLAR status into a reviewable artifact.
**Depends on**: Phase 91
**Requirements**: TRUST-01, TRUST-02, TRUST-03, TRUST-04
**Success Criteria** (what must be TRUE):
  1. Researcher can generate deterministic trust summary JSON and Markdown from existing sidecar/report refs.
  2. Trust summary clearly separates internally consistent, stable enough to interpret, evidence missing, diagnostic-only, and claim-upgrade-blocked outcomes.
  3. Trust summary references source reports by bounded refs and checksums rather than duplicating full payloads.
  4. Trust summary gives actionable next steps for CDNA3/MI300X/native-host/paper-scale validation while explicitly avoiding claims that v1.20 performed those validations.
**Plans**: TBD

### Phase 93: Documentation, Examples, And Guardrail Tests
**Goal**: Researchers can understand, reproduce, and safely interpret v1.20 evidence-quality artifacts through docs, fixtures, and guardrail tests.
**Depends on**: Phase 92
**Requirements**: DOCS-01, DOCS-02, DOCS-03, DOCS-04, DOCS-05
**Success Criteria** (what must be TRUE):
  1. Researcher can follow docs to generate and interpret consistency lint, evaluation stability, claim-upgrade, and trust summary artifacts.
  2. Documentation states that v1.20 does not add full 235-problem paper validation, CDNA3/MI300X/CDNA4 validation, native-host Matrix authority, hosted leaderboard readiness, or upstream SOLAR parity.
  3. CPU-safe tests cover contradiction detection, stability classification, claim-upgrade rejection, trust summary rendering, deterministic serialization, and docs claim-boundary wording.
  4. Public examples or fixtures show consistent, contradictory, noisy, and claim-blocked report shapes with bounded refs, checksums, and diagnostic-only wording.
  5. Existing public contracts remain stable: canonical Trace, Definition, Workload, Solution, correctness, timing, score, and evaluator contract semantics are unchanged.
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 89 -> 90 -> 91 -> 92 -> 93.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 89. Cross-Report Consistency Contract And Lint | 0/TBD | Not started | — |
| 90. Evaluation Stability Evidence | 0/TBD | Not started | — |
| 91. Claim Upgrade Rules And Authority Gates | 0/TBD | Not started | — |
| 92. Trust Summary Integration | 0/TBD | Not started | — |
| 93. Documentation, Examples, And Guardrail Tests | 0/TBD | Not started | — |

**Coverage:**
- Requirements mapped: 23/23
- Flow count: 5 phases
- Residual blockers: 0
