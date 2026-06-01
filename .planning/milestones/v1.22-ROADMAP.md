# Roadmap: SOL ExecBench ROCm Port

## Milestones

- Active **v1.22 Concern Closure and Execution Boundary Hardening** -
  Phases 100-105.

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

**Active milestone:** v1.22 Concern Closure and Execution Boundary Hardening.

**Status:** v1.22 phases complete.

**Next step:** Run milestone audit/completion when ready.

## Phases

- [x] **Phase 100: Dataset Runner Execution Seams** - Maintainers can change dataset execution, solution wrapping, reports, closure, and scheduling seams through importable helpers while preserving CLI behavior. (completed 2026-06-01)
- [x] **Phase 101: Eval Driver Diagnostics And Framing** - Maintainers can test reference timing, orchestration, trace emission, and output framing through package helpers without weakening benchmark compatibility. (completed 2026-06-01)
- [x] **Phase 102: Source Review And Boundary Evidence** - Maintainers and users get stronger source-review coverage and structured boundary evidence without implying hardened sandboxing. (completed 2026-06-01)
- [x] **Phase 103: Scoring And Static Evidence Fixtures** - Maintainers can validate SOLAR, AMD bound, and static-evidence behavior through family-specific fixtures and explicit artifact manifests. (completed 2026-06-01)
- [x] **Phase 104: Dependency And Closure Guardrails** - Maintainers can detect dependency-policy drift, closure-provenance regressions, and misleading hardware-marker outcomes. (completed 2026-06-01)
- [x] **Phase 105: Concern Map Stewardship** - Maintainers can use `CONCERNS.md` as an accurate milestone-management map for fixed, narrowed, carried-forward, and externally deferred concerns. (completed 2026-06-01)

## Phase Details

### Phase 100: Dataset Runner Execution Seams
**Goal**: Maintainers can evolve dataset-scale execution through importable runner helpers while `scripts/run_dataset.py` preserves existing user-facing behavior.
**Depends on**: Phase 99
**Requirements**: DATASET-01, DATASET-02, DATASET-03, DATASET-04
**Success Criteria** (what must be TRUE):
  1. Maintainer can invoke dataset problem execution through a package runner abstraction instead of embedding subprocess orchestration only in `scripts/run_dataset.py`.
  2. Maintainer can wrap reference and custom Python solutions without global source text replacement that mutates strings, comments, or legitimate identifiers.
  3. Dataset summaries, score reports, timing evidence refs, and closure reports are written through package helpers with focused tests.
  4. Dataset-scale CLI behavior remains compatible while exposing a safe seam for future scheduling or bounded CPU-side parallel report work.
**Plans**:
  - `100-01-PLAN.md` - Extract solution wrapping and subprocess invocation into package runner helpers.
  - `100-02-PLAN.md` - Extract summary, score, timing, and closure report seams while preserving CLI compatibility.

### Phase 101: Eval Driver Diagnostics And Framing
**Goal**: Maintainers can diagnose reference timing and output-framing behavior through importable helpers while preserving staged evaluator semantics.
**Depends on**: Phase 100
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):
  1. Maintainer can test reference timing behavior through package helpers without staging the full generated driver.
  2. When requested reference timing fails, traces, logs, or status semantics expose the failure explicitly instead of silently presenting `reference_latency_ms` as `0.0`.
  3. Regression tests prove user prints and noisy imports cannot corrupt trace JSONL output.
  4. Correctness and timing orchestration remains benchmark-compatible while avoidable pure logic moves out of `eval_driver.py`.
**Plans**:
  - `101-01-PLAN.md` - Extract reference timing diagnostics and noisy-output JSONL framing tests.

### Phase 102: Source Review And Boundary Evidence
**Goal**: Maintainers and users can see stronger static-review outcomes and boundary evidence without mistaking them for hard sandbox guarantees.
**Depends on**: Phase 101
**Requirements**: BOUNDARY-01, BOUNDARY-02, BOUNDARY-03, BOUNDARY-04
**Success Criteria** (what must be TRUE):
  1. Source-review tests cover additional process, file, import, native loader, stream, cache, and obfuscation bypass families.
  2. Python source review uses an AST-aware or token-aware path for cases where regex scanning is too broad or too easy to bypass.
  3. Blocked or flagged source-review outcomes appear as structured evidence in traces, sidecars, or logs.
  4. Public and developer docs state that static review plus subprocess execution is not hardened multi-tenant sandboxing.
**Plans**:
  - `102-01-PLAN.md` - Add AST-aware source review, structured boundary evidence, and sandbox-boundary docs.

### Phase 103: Scoring And Static Evidence Fixtures
**Goal**: Maintainers can validate scoring derivation and static-evidence changes against focused fixtures without changing public contracts or diagnostic authority.
**Depends on**: Phase 102
**Requirements**: SCORING-01, SCORING-02, SCORING-03, SCORING-04
**Success Criteria** (what must be TRUE):
  1. SOLAR and AMD bound derivation have family-specific golden fixtures for representative operator families and fallback behavior.
  2. Confidence and status transitions in SOLAR and AMD bound derivation are covered independently of broad report-shape tests.
  3. Static kernel evidence can consume or produce an explicit artifact manifest when build outputs are known.
  4. Static evidence and derived scoring changes preserve diagnostic-only authority and existing public sidecar contracts.
**Plans**:
  - `103-01-PLAN.md` - Add focused scoring fixtures and static artifact manifest support.

### Phase 104: Dependency And Closure Guardrails
**Goal**: Maintainers can catch policy, provenance, and marker regressions before they create misleading ROCm validation signals.
**Depends on**: Phase 103
**Requirements**: GUARD-01, GUARD-02, GUARD-03
**Success Criteria** (what must be TRUE):
  1. ROCm wheel, Docker target, and dependency-matrix policy consistency is guarded when target metadata changes.
  2. Dataset closure provenance tests cover new sidecar refs, stale provenance combinations, and manifest/cache provenance behavior.
  3. Hardware-marker skip behavior remains explicit so CPU-safe green runs cannot be mistaken for RDNA4, CDNA3, timing, MI300X, or CDNA4 validation.
**Plans**:
  - `104-01-PLAN.md` - Add dependency policy, closure provenance, and marker guardrail tests.

### Phase 105: Concern Map Stewardship
**Goal**: Maintainers can use `CONCERNS.md` as a reliable status map for v1.22 concern closure and future deferred work.
**Depends on**: Phase 104
**Requirements**: DOCS-01, DOCS-02, DOCS-03
**Success Criteria** (what must be TRUE):
  1. `CONCERNS.md` preserves milestone-management context for recently narrowed, still actionable, accepted, and externally deferred concerns.
  2. v1.22 completion updates each in-scope concern as fixed, narrowed, or carried forward with evidence.
  3. Out-of-scope items remain explicit: CDNA3, MI300X, CDNA4 validation, paper-scale parity, leaderboard readiness, and complete hard sandboxing.
**Plans**:
  - `105-01-PLAN.md` - Update CONCERNS.md with v1.22 outcomes and deferred boundaries.

## Progress

**Execution Order:**
Phases execute in numeric order: 100 -> 101 -> 102 -> 103 -> 104 -> 105.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 100. Dataset Runner Execution Seams | 2/2 | Complete    | 2026-06-01 |
| 101. Eval Driver Diagnostics And Framing | 1/1 | Complete    | 2026-06-01 |
| 102. Source Review And Boundary Evidence | 1/1 | Complete    | 2026-06-01 |
| 103. Scoring And Static Evidence Fixtures | 1/1 | Complete    | 2026-06-01 |
| 104. Dependency And Closure Guardrails | 1/1 | Complete    | 2026-06-01 |
| 105. Concern Map Stewardship | 1/1 | Complete    | 2026-06-01 |

**Coverage:**
- Requirements mapped: 22/22
- Requirements complete: 0/22
- Flow count: 6 phases
- Residual blockers: 0
