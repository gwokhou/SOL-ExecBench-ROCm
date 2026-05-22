# Roadmap: SOL ExecBench ROCm Port

## Current Milestone: v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness

**Goal:** Adapt selected `hip-execbench` engineering practices without public
contract breakage, implement CDNA 3 validation readiness without a hardware
validation claim, and validate the implemented RDNA 4 path with unit and E2E
evidence.

**Phase numbering:** Continues from v1.3. v1.4 starts at Phase 19.

## Phase Summary

| Phase | Name | Goal | Requirements |
|-------|------|------|--------------|
| 19 | Compatibility and Practice Inventory | Complete 2026-05-22: Established non-negotiable contracts and classified `hip-execbench` practices before implementation. | COMPAT-01, COMPAT-02, COMPAT-03 |
| 20 | Internal Diagnostics and Evidence Model | Complete 2026-05-22: Added derived evidence/report helpers without changing benchmark output contracts. | ENG-04, ENG-05, ENG-06 |
| 21 | CDNA 3 Validation Readiness | Complete 2026-05-22: Implemented `gfx94*` readiness metadata, evidence requirements, blockers, and no-claim guardrails without CDNA 3 hardware validation. | VAL-04, VAL-05, VAL-06 |
| 22 | RDNA 4 Validation Closure | Complete 2026-05-22: Validated v1.4 on RDNA 4 with unit, E2E pytest, and `sol-execbench` CLI trace evidence. | RDNA-01, RDNA-02, RDNA-03 |

**Coverage:** 12 / 12 v1.4 requirements mapped.

## Phases

### Phase 19: Compatibility and Practice Inventory

**Status:** Complete 2026-05-22

**Goal:** Establish the v1.4 compatibility boundary and source-grounded
`hip-execbench` adaptation decisions before touching implementation paths.

**Requirements:** COMPAT-01, COMPAT-02, COMPAT-03

**Success criteria:**

1. Public contracts are inventoried with explicit references to current CLI,
   schema, trace JSONL, solution format, and eval-driver behavior.
2. Guardrail tests protect the inventoried contracts from unintentional v1.4
   drift.
3. `hip-execbench` practices are classified as accepted, rejected, or deferred
   with rationale tied to compatibility and benchmark semantics.
4. No runtime dependency or public API change is introduced in this phase.

**Implementation notes:**

- Extend existing public-contract guardrail test style.
- Use local source evidence from both repositories; do not rely on README
  claims as implementation evidence.

### Phase 20: Internal Diagnostics and Evidence Model

**Status:** Complete 2026-05-22

**Goal:** Adapt `hip-execbench` stage-result and report/evidence practices as
internal or derived helpers while preserving trace JSONL as the canonical
benchmark output.

**Requirements:** ENG-04, ENG-05, ENG-06

**Success criteria:**

1. Maintainers can inspect parse/package/compile/evaluate/report-style
   readiness through internal diagnostics or evidence data structures.
2. Derived reports or evidence artifacts are generated from existing traces and
   diagnostics without mutating trace JSONL.
3. Any agent-readable or report-style output labels itself as derived and not
   canonical benchmark output.
4. Tests prove existing public trace schema and CLI behavior remain unchanged.

**Implementation notes:**

- Prefer dataclasses and pure helpers under `src/sol_execbench/core/`.
- Avoid adding mandatory runtime dependencies.
- Avoid replacing `ProblemPackager`, `build_ext.py`, or `eval_driver.py`.

### Phase 21: CDNA 3 Validation Readiness

**Status:** Complete 2026-05-22

**Goal:** Implement CDNA 3 validation readiness for `gfx94*` so a future real
hardware run has commands, evidence requirements, blockers, and acceptance
criteria ready, while explicitly avoiding a validation-pass claim.

**Requirements:** VAL-04, VAL-05, VAL-06

**Success criteria:**

1. Readiness logic distinguishes RDNA 4, CDNA 3 `gfx94*`, unknown AMD targets,
   and missing ROCm hardware/tooling.
2. CDNA 3 readiness output identifies expected commands, evidence files,
   acceptance criteria, and blockers for a future real `gfx94*` run.
3. Docs and tests explicitly distinguish readiness from real CDNA 3 hardware
   validation.
4. Focused unit tests cover readiness behavior without requiring real CDNA 3
   hardware.

**Implementation notes:**

- Reuse and extend existing ROCm diagnostics patterns where possible.
- Keep claim wording conservative: "readiness implemented" is not "validated".

### Phase 22: RDNA 4 Validation Closure

**Status:** Complete 2026-05-22

**Goal:** Validate v1.4 implementation on RDNA 4 with unit and E2E evidence and
prove compatibility and benchmark semantics remain intact.

**Requirements:** RDNA-01, RDNA-02, RDNA-03

**Success criteria:**

1. Relevant unit tests for v1.4 diagnostics, evidence, readiness, and
   compatibility guardrails pass on RDNA 4.
2. Existing `sol-execbench` benchmark flow is exercised on RDNA 4 and produces
   valid trace output.
3. Final validation confirms reference correctness, timing integrity,
   reward-hack defenses, ROCm-only schema/build/eval behavior, and public
   compatibility guardrails did not regress.
4. Validation evidence is recorded in a durable planning or documentation
   artifact for milestone audit.

**Implementation notes:**

- Use existing E2E samples and markers unless a new focused sample is necessary.
- Do not claim CDNA 3 validation during this phase.

## Completed Milestones

- [x] **v1.3 Non-CDNA Issue Closure** — shipped 2026-05-22. See `.planning/milestones/v1.3-ROADMAP.md`.
- [x] **v1.2 Engineering Practice Harvest and Compatibility Guardrails** — shipped 2026-05-22. See `.planning/milestones/v1.2-ROADMAP.md`.
- [x] **v1.1 CDNA 3 Support and Migration Closure** — shipped 2026-05-21. See `.planning/milestones/v1.1-ROADMAP.md`.
- [x] **v1.0 ROCm Port** — shipped 2026-05-21. See `.planning/milestones/v1.0-ROADMAP.md`.

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| COMPAT-01 | Phase 19 | Complete |
| COMPAT-02 | Phase 19 | Complete |
| COMPAT-03 | Phase 19 | Complete |
| ENG-04 | Phase 20 | Complete |
| ENG-05 | Phase 20 | Complete |
| ENG-06 | Phase 20 | Complete |
| VAL-04 | Phase 21 | Complete |
| VAL-05 | Phase 21 | Complete |
| VAL-06 | Phase 21 | Complete |
| RDNA-01 | Phase 22 | Complete |
| RDNA-02 | Phase 22 | Complete |
| RDNA-03 | Phase 22 | Complete |
