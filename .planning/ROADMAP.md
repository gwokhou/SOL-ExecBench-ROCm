# Roadmap: SOL ExecBench ROCm Port

## Current Milestone: v1.5 AMD-native SOL Scoring and ROCm Profiler Timing

**Goal:** Build a SOLAR-like AMD-native scoring foundation and replace default
ROCm timing with an accuracy-first profiler-backed timing model, while keeping
real CDNA3 hardware validation explicitly out of scope.

**Phase numbering:** Continues from v1.4. v1.5 starts at Phase 23.

## Phase Summary

| Phase | Name | Goal | Requirements |
|-------|------|------|--------------|
| 23 | Timing Semantics and Policy | Complete 2026-05-22: Defined source classification and accuracy-first timing policy for HIP native, Triton, PyTorch, and mixed workloads. | TIME-01, TIME-02, TIME-03, TIME-04 |
| 24 | rocprofv3 Default Timing Path | Complete 2026-05-22: Added profiler-backed ROCm timing evidence helpers, parser fixtures, policy-aware default selection, and labeled fallbacks. | PROF-01, PROF-02, PROF-03, PROF-04 |
| 25 | AMD SOL Bound Foundation | Complete 2026-05-22: Added conservative graph extraction, FLOP/byte estimates, AMD hardware model metadata, and derived bound artifacts. | SOL-01, SOL-02, SOL-03, SOL-04 |
| 26 | AMD-native Scoring and Guarded Reports | Produce AMD-native per-problem and suite scores with baseline comparison, evidence references, compatibility guardrails, and CDNA3 no-claim protection. | SCORE-01, SCORE-02, SCORE-03, SCORE-04, COMPAT-01, COMPAT-02, CLAIM-01, CLAIM-02 |

**Coverage:** 20 / 20 v1.5 requirements mapped.

## Phases

### Phase 23: Timing Semantics and Policy

**Status:** Complete 2026-05-22

**Goal:** Define how benchmark work is classified and timed before changing the
default timing implementation.

**Requirements:** TIME-01, TIME-02, TIME-03, TIME-04

**Success criteria:**

1. Maintainers can classify measured work as HIP native, Triton, PyTorch, or
   mixed source type before a timing backend is selected.
2. A timing policy table maps each source type to timer backend and
   interpretation.
3. The implementation permits source-specific timing chimneys when a unified
   timing口径 would reduce accuracy.
4. Documentation explains whether each timing path measures kernel activity,
   HIP runtime/API activity, PyTorch operator attribution, or fallback event
   timing.

**Implementation notes:**

- Keep timing accuracy as the top rule.
- Start with pure policy/data models and tests before invoking external
  profilers.
- Treat PyTorch ROCm's `torch.cuda` and `ProfilerActivity.CUDA` naming as API
  compatibility surface, not NVIDIA runtime evidence.

**Plans:** 1 plan

Plans:

- [x] 23-01: Add timing source classification, timing policy models, chimney
  documentation, and focused policy/audit tests.

### Phase 24: rocprofv3 Default Timing Path

**Status:** Complete 2026-05-22

**Goal:** Implement profiler-backed ROCm timing and make it the default timing
path when it is the most accurate supported backend for the classified source
type.

**Requirements:** PROF-01, PROF-02, PROF-03, PROF-04

**Success criteria:**

1. Maintainers can collect and parse representative `rocprofv3` timing evidence
   for benchmark executions.
2. The default benchmark timing path uses profiler-backed timing where the
   Phase 23 policy selects it.
3. Fallback timing paths are explicitly labeled with backend, reason, and
   interpretation.
4. Timing evidence records tool version, GPU architecture, activity domain,
   aggregation rule, and parsed rows needed to audit measured duration.

**Implementation notes:**

- Add fixture tests for profiler output parsing before relying on local hardware
  traces.
- Keep profiler output in controlled evidence directories.
- Do not include Triton JIT/autotune or PyTorch setup overhead in steady-state
  device timing unless the evidence explicitly labels that interpretation.

**Plans:** 1 plan

Plans:

- [x] 24-01: Add rocprofv3 command construction, CSV parser, timing evidence,
  policy-aware default selection, fallback metadata, and profiler evidence docs.

### Phase 25: AMD SOL Bound Foundation

**Status:** Complete 2026-05-22

**Goal:** Build a SOLAR-like AMD bound pipeline that produces auditable
theoretical bound artifacts before scoring.

**Requirements:** SOL-01, SOL-02, SOL-03, SOL-04

**Success criteria:**

1. Maintainers can run graph extraction for supported benchmark workloads.
2. FLOP and byte analysis records supported, inexact, and unsupported
   operations with confidence and rationale.
3. AMD hardware model entries record architecture, dtype or execution path,
   peak-value source, confidence, and validation status.
4. Per-op and aggregate AMD SOL bound artifacts are generated before
   AMD-native scoring is allowed.

**Implementation notes:**

- Prefer evidence-carrying bound objects over bare score formulas.
- Keep hardware model data versioned and auditable.
- CDNA3 model scaffolding is allowed only with unvalidated claim status.

**Plans:** 1 plan

Plans:

- [x] 25-01: Add AMD SOL graph extraction, FLOP/byte estimates, hardware model
  metadata, bound artifact generation, and documentation guardrails.

### Phase 26: AMD-native Scoring and Guarded Reports

**Status:** Not started

**Goal:** Combine timing evidence, AMD SOL bounds, baselines, and claim
guardrails into an end-to-end AMD-native scoring workflow.

**Requirements:** SCORE-01, SCORE-02, SCORE-03, SCORE-04, COMPAT-01,
COMPAT-02, CLAIM-01, CLAIM-02

**Success criteria:**

1. Maintainers can generate per-problem AMD-native scores from measured timing
   and AMD SOL bound artifacts.
2. Baseline ingestion and comparison work without claiming NVIDIA B200, SOLAR,
   or leaderboard equivalence.
3. Suite-level aggregation preserves references to each workload's timing and
   bound evidence.
4. Score and report outputs include guardrails for unsupported, incomplete, or
   unvalidated evidence.
5. Existing CLI behavior, solution schema, canonical trace JSONL, eval-driver
   correctness semantics, and reward-hack defenses do not regress.
6. Reports explicitly state that real CDNA3 `gfx94*` full-suite validation is
   excluded from v1.5.

**Implementation notes:**

- Keep SOL and timing evidence as derived artifacts unless an additive,
  documented output path is explicitly introduced.
- Reuse existing scoring guardrail patterns where they fit.
- Add compatibility tests around trace/schema behavior before final closure.

## Completed Milestones

- [x] **v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness** - shipped 2026-05-22. See `.planning/milestones/v1.4-ROADMAP.md`.
- [x] **v1.3 Non-CDNA Issue Closure** - shipped 2026-05-22. See `.planning/milestones/v1.3-ROADMAP.md`.
- [x] **v1.2 Engineering Practice Harvest and Compatibility Guardrails** - shipped 2026-05-22. See `.planning/milestones/v1.2-ROADMAP.md`.
- [x] **v1.1 CDNA 3 Support and Migration Closure** - shipped 2026-05-21. See `.planning/milestones/v1.1-ROADMAP.md`.
- [x] **v1.0 ROCm Port** - shipped 2026-05-21. See `.planning/milestones/v1.0-ROADMAP.md`.

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TIME-01 | Phase 23 | Complete |
| TIME-02 | Phase 23 | Complete |
| TIME-03 | Phase 23 | Complete |
| TIME-04 | Phase 23 | Complete |
| PROF-01 | Phase 24 | Complete |
| PROF-02 | Phase 24 | Complete |
| PROF-03 | Phase 24 | Complete |
| PROF-04 | Phase 24 | Complete |
| SOL-01 | Phase 25 | Complete |
| SOL-02 | Phase 25 | Complete |
| SOL-03 | Phase 25 | Complete |
| SOL-04 | Phase 25 | Complete |
| SCORE-01 | Phase 26 | Pending |
| SCORE-02 | Phase 26 | Pending |
| SCORE-03 | Phase 26 | Pending |
| SCORE-04 | Phase 26 | Pending |
| COMPAT-01 | Phase 26 | Pending |
| COMPAT-02 | Phase 26 | Pending |
| CLAIM-01 | Phase 26 | Pending |
| CLAIM-02 | Phase 26 | Pending |
