# Roadmap: SOL ExecBench ROCm Port

## Milestones

- Complete **v1.24 Dataset Batch Run Trustworthiness** - Phases 110-113
  (shipped 2026-06-01). See `.planning/milestones/v1.24-ROADMAP.md`.

- Complete **v1.23 Evaluation Reliability and Security Hardening** -
  Phases 106-109 (shipped 2026-06-01). See
  `.planning/milestones/v1.23-ROADMAP.md`.

- Complete **v1.22 Concern Closure and Execution Boundary Hardening** -
  Phases 100-105 (shipped 2026-06-01). See
  `.planning/milestones/v1.22-ROADMAP.md`.

- Complete **v1.21 Codebase Debt Reduction and Execution Boundary
  Hardening** - Phases 94-99 (shipped 2026-06-01). See
  `.planning/milestones/v1.21-ROADMAP.md` when archived.

- Complete **v1.20 Cross-Report Consistency and Evaluation Stability** -
  Phases 89-93 (shipped 2026-05-31). See
  `.planning/milestones/v1.20-ROADMAP.md`.

- Complete **v1.19 Research Credibility Without New Hardware** -
  Phases 83-88 (shipped 2026-05-31). See
  `.planning/milestones/v1.19-ROADMAP.md`.

- Earlier milestones v1.0-v1.18 are archived under `.planning/milestones/`.

## Current Position

**Active milestone:** None.

**Status:** v1.23 and v1.24 complete.

**Next step:** Define the next milestone when ready.

## Phases

- [x] **Phase 106: Evaluation Failure Diagnostics** - Users and
  maintainers can diagnose no-trace and noisy-output failures through bounded
  stdout/stderr evidence without relying on verbose-only console output.
  (completed 2026-06-01)
- [x] **Phase 107: Staged User Import Isolation** - Python submissions are
  loaded through unique file-based module identities so staged solution names
  cannot collide with already-imported driver or dependency modules.
  (completed 2026-06-01)
- [x] **Phase 108: Native Compile Option Guardrails** - Native solution
  compiler and linker options reject dangerous host/path/linker behavior while
  preserving documented ROCm/HIP extension use cases.
  (completed 2026-06-01)
- [x] **Phase 109: Eval Driver Responsibility Boundaries** - The generated
  eval driver delegates benchmark phases to tested importable helpers while
  preserving canonical trace, correctness, timing, and reward-hack semantics.
  (completed 2026-06-01)
- [x] **Phase 110: Dataset Reuse Policy Service** - Dataset batch reuse and
  stale-provenance decisions are owned by tested package services instead of
  script-local branching.
  (completed 2026-06-01)
- [x] **Phase 111: Dataset Closure And Evidence Completeness** - Dataset
  closure records, timing refs, derived sidecars, and missing-evidence states
  are constructed through validated core helpers.
  (completed 2026-06-01)
- [x] **Phase 112: Dataset Failure-Mode Regression Matrix** - Stale
  provenance, selected ready-subset workloads, rerun flags, missing sidecars,
  and nonzero CLI outcomes have focused regression coverage.
  (completed 2026-06-01)
- [x] **Phase 113: Deterministic Dataset Sharding Path** - Dataset-scale
  execution has a first deterministic shard and merge design that preserves
  one trace file per shard and explicit merge rules.
  (completed 2026-06-01)

## Phase Details

### Phase 106: Evaluation Failure Diagnostics
**Goal**: Users and maintainers can diagnose no-trace, noisy-output, and
nonzero evaluation failures through bounded persisted diagnostics.
**Depends on**: Phase 105
**Requirements**: EVAL-DIAG-01, EVAL-DIAG-02, EVAL-DIAG-03
**Success Criteria** (what must be TRUE):
  1. No-trace evaluation outcomes persist bounded stdout and stderr sidecars or
     equivalent structured diagnostic evidence.
  2. CLI failure reporting points to the diagnostic evidence without requiring
     `--verbose` or manual staging inspection.
  3. Regression tests cover non-JSON stdout, library noise, nonzero exits, and
     empty trace output without changing canonical trace JSONL.
**Plans**:
  - `106-01-PLAN.md` - Add bounded no-trace diagnostics and CLI failure
    reporting.

### Phase 107: Staged User Import Isolation
**Goal**: Python solution imports use unique staged module identities and do
not depend on dotted names that can collide with `sys.modules`.
**Depends on**: Phase 106
**Requirements**: EVAL-IMPORT-01, EVAL-IMPORT-02, EVAL-IMPORT-03
**Success Criteria** (what must be TRUE):
  1. `load_user_function()` imports Python sources through a file-based
     `importlib` path with a unique generated module name.
  2. Package-style and simple-file solution entries keep their supported
     behavior while avoiding collisions with previously imported modules.
  3. Regression tests prove collisions such as `main.py::run` and package-like
     module names cannot resolve to unintended existing modules.
**Plans**:
  - `107-01-PLAN.md` - Replace dotted staged imports with unique file-based
    imports.

### Phase 108: Native Compile Option Guardrails
**Goal**: Native solution compile options are constrained enough for trusted
research evaluation while remaining compatible with documented ROCm extension
builds.
**Depends on**: Phase 107
**Requirements**: COMPILE-GUARD-01, COMPILE-GUARD-02, COMPILE-GUARD-03
**Success Criteria** (what must be TRUE):
  1. Dangerous compiler/linker options that reference host paths, dynamic
     loaders, response files, or unsafe link-time behavior are rejected with
     clear validation errors.
  2. Allowed ROCm/HIP compile options used by examples and tests remain
     accepted.
  3. Schema and build-template tests cover accepted and rejected option
     classes.
**Plans**:
  - `108-01-PLAN.md` - Add native compile option validation and tests.

### Phase 109: Eval Driver Responsibility Boundaries
**Goal**: The generated driver becomes a thinner staged entry point while
benchmark behavior remains compatible.
**Depends on**: Phase 108
**Requirements**: EVAL-BOUNDARY-01, EVAL-BOUNDARY-02, EVAL-BOUNDARY-03,
EVAL-BOUNDARY-04
**Success Criteria** (what must be TRUE):
  1. Correctness, timing, trace emission, source review, and reward-hack
     boundary behavior are exposed through importable helpers with focused
     tests.
  2. The generated template remains responsible only for staged wiring,
     subprocess-local setup, and integration glue.
  3. Integrity snapshot coverage remains explicit for every benchmark-critical
     helper reachable from staged execution.
  4. Public trace, correctness, timing, score, and evaluator contract schemas
     remain unchanged unless a separate schema discussion approves a change.
**Plans**:
  - `109-01-PLAN.md` - Thin generated eval driver around tested benchmark
    phase helpers.

### Phase 110: Dataset Reuse Policy Service
**Goal**: Dataset reuse and stale-provenance decisions are centralized in
tested core services.
**Depends on**: Phase 109
**Requirements**: DATASET-REUSE-01, DATASET-REUSE-02, DATASET-REUSE-03
**Success Criteria** (what must be TRUE):
  1. Reuse decisions are computed by importable helpers with explicit inputs
     for selected workloads, ready subsets, provenance, rerun flags, and
     output paths.
  2. `scripts/run_dataset.py` delegates policy decisions instead of owning the
     decision matrix inline.
  3. Tests cover reuse, stale provenance, forced rerun, partial-ready, and
     missing-output combinations.
**Plans**:
  - `110-01-PLAN.md` - Extract dataset reuse policy into core helpers.

### Phase 111: Dataset Closure And Evidence Completeness
**Goal**: Dataset closure and evidence references are assembled through core
helpers with explicit incomplete-state semantics.
**Depends on**: Phase 110
**Requirements**: DATASET-CLOSURE-01, DATASET-CLOSURE-02, DATASET-CLOSURE-03
**Success Criteria** (what must be TRUE):
  1. Closure records classify missing traces, missing timing evidence, missing
     derived sidecars, skipped workloads, and nonzero CLI outcomes without
     ambiguous success states.
  2. Summary, score, timing, and closure references use package-level helpers
     rather than script-local path coupling.
  3. Closure reports remain deterministic and compatible with existing public
     sidecar contracts.
**Plans**:
  - `111-01-PLAN.md` - Centralize dataset closure and evidence completeness
    helpers.

### Phase 112: Dataset Failure-Mode Regression Matrix
**Goal**: Dataset runner failure modes have targeted tests that prevent silent
stale evidence or incomplete closure reporting.
**Depends on**: Phase 111
**Requirements**: DATASET-REGRESS-01, DATASET-REGRESS-02, DATASET-REGRESS-03
**Success Criteria** (what must be TRUE):
  1. Regression fixtures cover stale provenance, selected ready subsets,
     missing derived sidecars, rerun flags, CLI timeout/nonzero outcomes, and
     no-trace outputs.
  2. Tests distinguish CPU-safe policy behavior from live ROCm/GPU execution
     requirements.
  3. Dataset runner documentation explains when reuse is allowed, blocked, or
     reported as incomplete.
**Plans**:
  - `112-01-PLAN.md` - Add dataset failure-mode regression coverage and docs.

### Phase 113: Deterministic Dataset Sharding Path
**Goal**: Dataset-scale runs have a deterministic sharding design that can
reduce single-process pressure without weakening trace provenance.
**Depends on**: Phase 112
**Requirements**: DATASET-SHARD-01, DATASET-SHARD-02, DATASET-SHARD-03
**Success Criteria** (what must be TRUE):
  1. A shard plan can divide workloads deterministically with stable shard
     identifiers and one trace file per shard.
  2. Merge rules preserve workload ordering, provenance, duplicate detection,
     and incomplete-shard reporting.
  3. The first implementation or design artifact is covered by tests and keeps
     default dataset CLI behavior compatible.
**Plans**:
  - `113-01-PLAN.md` - Define and test deterministic dataset shard and merge
    semantics.

## Progress

**Execution Order:**
Phases execute in numeric order: 106 -> 107 -> 108 -> 109, then queued
milestone phases 110 -> 111 -> 112 -> 113.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 106. Evaluation Failure Diagnostics | 1/1 | Complete | 2026-06-01 |
| 107. Staged User Import Isolation | 1/1 | Complete | 2026-06-01 |
| 108. Native Compile Option Guardrails | 1/1 | Complete | 2026-06-01 |
| 109. Eval Driver Responsibility Boundaries | 1/1 | Complete | 2026-06-01 |
| 110. Dataset Reuse Policy Service | 1/1 | Complete | 2026-06-01 |
| 111. Dataset Closure And Evidence Completeness | 1/1 | Complete | 2026-06-01 |
| 112. Dataset Failure-Mode Regression Matrix | 1/1 | Complete | 2026-06-01 |
| 113. Deterministic Dataset Sharding Path | 1/1 | Complete | 2026-06-01 |

**Coverage:**
- v1.23 requirements complete: 13/13
- v1.24 requirements complete: 12/12
- Completed milestone flow count: 8 phases
- Residual blockers: 0
