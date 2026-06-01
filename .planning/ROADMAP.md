# Roadmap: SOL ExecBench ROCm Port

## Milestones

- Active **v1.21 Codebase Debt Reduction and Execution Boundary Hardening** -
  Phases 94-99.

- Complete **v1.20 Cross-Report Consistency and Evaluation Stability** -
  Phases 89-93 (shipped 2026-05-31). See
  `.planning/milestones/v1.20-ROADMAP.md`.

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

- Earlier milestones v1.0-v1.15 are archived under `.planning/milestones/`.

## Current Position

**Active milestone:** v1.21 Codebase Debt Reduction and Execution Boundary
Hardening.

**Status:** requirements and roadmap defined; ready for Phase 94 planning.

**Next step:** Run `$gsd-discuss-phase 94` or `$gsd-plan-phase 94`.

## Phases

- [ ] **Phase 94: Dataset Runner Decomposition** - Maintainers can modify dataset execution, resume, closure, and derived evidence behavior through tested package helpers instead of a monolithic script.
- [ ] **Phase 95: Eval Driver Runtime Decomposition** - Maintainers can test more evaluator behavior in package modules while preserving staged template smoke coverage.
- [ ] **Phase 96: AMD Bound Graph And Estimate Modularization** - Maintainers can change AMD bound graph extraction and estimate formulas by operation family with focused tests.
- [ ] **Phase 97: SOLAR Derivation And Static Evidence Modularization** - Maintainers can work on SOLAR derivation and static evidence through isolated provenance, parser, status, and rendering helpers.
- [ ] **Phase 98: Execution Boundary Test Hardening** - Known fragile areas gain CPU-safe and ROCm-aware tests for reward-hack, clock/timing, static evidence, and dataset closure edge cases.
- [ ] **Phase 99: Boundary Documentation And Final Concern Closure** - Public docs, guardrails, and `CONCERNS.md` accurately separate fixed debt, narrowed risks, accepted limits, and future external work.

## Phase Details

### Phase 94: Dataset Runner Decomposition
**Goal**: Maintainers can reason about dataset execution behavior through small, tested package helpers while `scripts/run_dataset.py` becomes orchestration-only.
**Depends on**: Phase 93
**Requirements**: DATASET-01, DATASET-02, DATASET-03, DATASET-04, DATASET-05
**Success Criteria** (what must be TRUE):
  1. Dataset selection and workload limiting behavior is delegated to package helpers with focused tests.
  2. Resume/rerun/existing-output decisions are deterministic and covered for capped workloads, ready subsets, stale traces, and stale closure provenance.
  3. Closure record construction preserves existing bounded refs, status vocabulary, and failure classifications.
  4. Derived evidence discovery can run independently from the main execution loop and attach refs without changing public sidecar shapes.
  5. Existing run_dataset CLI contracts and representative tests continue to pass.
**Plans**:
  - [ ] 94-01 Selection And Run-State Helpers
  - [ ] 94-02 Closure And Derived Evidence Helpers

### Phase 95: Eval Driver Runtime Decomposition
**Goal**: Maintainers can unit-test more eval-driver behavior before staging while the generated template remains the subprocess integration shell.
**Depends on**: Phase 94
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):
  1. Additional deterministic evaluator helpers move into importable runtime modules with focused unit tests.
  2. The template retains staging-directory dynamic imports and trace emission glue but loses avoidable inline pure logic.
  3. Reward-hack check plumbing and evaluation construction preserve status priority and log behavior.
  4. Driver smoke tests continue to cover passing, invalid reference, reward-hack, runtime error, and template syntax paths.
**Plans**:
  - [ ] 95-01 Runtime Helper Extraction
  - [ ] 95-02 Template Smoke And Status Guardrails

### Phase 96: AMD Bound Graph And Estimate Modularization
**Goal**: Maintainers can modify AMD bound graph and estimate behavior by operation family without broad unrelated scoring regressions.
**Depends on**: Phase 95
**Requirements**: ANALYSIS-01, ANALYSIS-02
**Success Criteria** (what must be TRUE):
  1. Graph construction, operator-family classification, and evidence annotation responsibilities are separated behind stable helpers.
  2. Estimate formulas are grouped by family or responsibility with tests covering representative elementwise, reduction, matmul, attention, and fallback behavior where applicable.
  3. Existing AMD bound graph and estimate public outputs remain schema-compatible.
  4. Scoring tests prove no unintended authority or claim-boundary changes.
**Plans**:
  - [ ] 96-01 Bound Graph Responsibility Split
  - [ ] 96-02 Estimate Formula Family Split

### Phase 97: SOLAR Derivation And Static Evidence Modularization
**Goal**: Maintainers can work on SOLAR derivation and static evidence through smaller provenance, parser, status, and rendering units.
**Depends on**: Phase 96
**Requirements**: ANALYSIS-03, ANALYSIS-04, ANALYSIS-05
**Success Criteria** (what must be TRUE):
  1. SOLAR derivation separates semantic provenance, bound/formula derivation, coverage/status classification, and report rendering.
  2. Static evidence separates artifact discovery, tool routing, bounded output capture, parser behavior, and sidecar/report rendering.
  3. Parser and status fixtures cover available, unavailable, failed, partial, and toolchain-variant states.
  4. Existing SOLAR/static evidence sidecar schemas and diagnostic-only authority boundaries are preserved.
**Plans**:
  - [ ] 97-01 SOLAR Derivation Responsibility Split
  - [ ] 97-02 Static Evidence Responsibility Split

### Phase 98: Execution Boundary Test Hardening
**Goal**: Fragile execution-boundary areas have stronger CPU-safe and ROCm-aware regression coverage without claiming hard sandboxing or new hardware validation.
**Depends on**: Phase 97
**Requirements**: BOUNDARY-01, BOUNDARY-02, BOUNDARY-03, BOUNDARY-04
**Success Criteria** (what must be TRUE):
  1. Reward-hack catalog tests cover additional known bypass spellings and preserve allowed-case tests for intentional false positives.
  2. Clock/timing tests include representative ROCm SMI/device fixture outputs, low-power/unsupported states, and memory/timing diagnostics where hardware is unavailable.
  3. Static evidence tests cover partial, unavailable, failed, and parser-sensitive toolchain outputs through fixtures.
  4. Dataset resume/closure tests cover stale traces, stale closure provenance, capped workloads, ready subsets, reruns, missing traces, and derived evidence combinations.
**Plans**:
  - [ ] 98-01 Reward-Hack And Clock Timing Coverage
  - [ ] 98-02 Static Evidence And Dataset Closure Coverage

### Phase 99: Boundary Documentation And Final Concern Closure
**Goal**: Researchers and maintainers can see which `CONCERNS.md` items were fixed, narrowed, accepted, or deferred to external evidence/runner work.
**Depends on**: Phase 98
**Requirements**: BOUNDARY-05, DOCS-01, DOCS-02, DOCS-03, DOCS-04
**Success Criteria** (what must be TRUE):
  1. `CONCERNS.md` is updated with current status for every v1.21-targeted concern.
  2. Public docs clearly state that v1.21 does not add hard sandboxing, multi-tenant safety, CDNA3/MI300X validation, paper-scale parity, or leaderboard authority.
  3. Guardrail tests prevent diagnostic evidence, Docker evidence, local AMD SOL/SOLAR interpretations, or static evidence from becoming stronger public claims.
  4. Developer docs explain new module boundaries for dataset execution, eval driver runtime, scoring derivation, and static evidence.
  5. Final milestone audit can verify all v1.21 requirements are mapped, tested, and boundary-safe.
**Plans**:
  - [ ] 99-01 Docs And Claim Guardrails
  - [ ] 99-02 Concern Closure And Milestone Audit Prep

## Progress

**Execution Order:**
Phases execute in numeric order: 94 -> 95 -> 96 -> 97 -> 98 -> 99.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 94. Dataset Runner Decomposition | 0/2 | Planned | — |
| 95. Eval Driver Runtime Decomposition | 0/2 | Planned | — |
| 96. AMD Bound Graph And Estimate Modularization | 0/2 | Planned | — |
| 97. SOLAR Derivation And Static Evidence Modularization | 0/2 | Planned | — |
| 98. Execution Boundary Test Hardening | 0/2 | Planned | — |
| 99. Boundary Documentation And Final Concern Closure | 0/2 | Planned | — |

**Coverage:**
- Requirements mapped: 23/23
- Requirements complete: 0/23
- Flow count: 6 phases
- Residual blockers: 0
