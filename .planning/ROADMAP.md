# Roadmap: SOL ExecBench ROCm Port

## Current Milestone: v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow

**Goal:** Turn the v1.5 AMD-native SOL, timing, and scoring foundations into an
end-to-end workflow while preserving existing public contracts.

**Phase numbering:** Continues from v1.5. v1.6 starts at Phase 27.

## Phase Summary

| Phase | Name | Goal | Requirements |
|-------|------|------|--------------|
| 27 | AMD SOL Analyzer Coverage | Broaden AMD SOL/SOLAR-like analyzer coverage and make coverage confidence visible before scoring. | SOLCOV-01, SOLCOV-02, SOLCOV-03, SOLCOV-04 |
| 28 | Live rocprofv3 Timing Integration | Collect live source-specific ROCm profiler timing evidence during benchmark or dataset execution. | PROF-01, PROF-02, PROF-03, PROF-04 |
| 29 | Derived AMD Scoring Workflow | Connect trace JSONL, live timing evidence, AMD SOL bounds, and baseline inputs into workload and suite reports. | SCORE-01, SCORE-02, SCORE-03, SCORE-04 |
| 30 | Compatibility and Claim Guardrails | Prove public contracts remain compatible and keep CDNA3/NVIDIA equivalence claims out of v1.6. | COMPAT-01, COMPAT-02, COMPAT-03, COMPAT-04, CLAIM-01, CLAIM-02, CLAIM-03 |

**Coverage:** 19 / 19 v1.6 requirements mapped.

## Phases

### Phase 27: AMD SOL Analyzer Coverage

**Status:** Pending

**Goal:** Broaden AMD SOL/SOLAR-like analyzer coverage and make coverage
confidence visible before scoring.

**Requirements:** SOLCOV-01, SOLCOV-02, SOLCOV-03, SOLCOV-04

**Success criteria:**

1. Maintainers can analyze representative workloads with operator coverage
   beyond v1.5 matmul and broad elementwise detection.
2. Per-operation analyzer output labels each node as supported, inexact, or
   unsupported with rationale.
3. A derived coverage summary reports supported, inexact, and unsupported
   counts before score generation.
4. AMD SOL bound artifacts keep per-op FLOP, byte, limiting-resource,
   confidence, and hardware-model evidence separate from canonical trace JSONL.

**Implementation notes:**

- Prefer an analyzer registry or similarly explicit structure over expanding a
  long AST string-check chain.
- Prioritize correctness of confidence labels over broad unsupported estimates.
- Keep unsupported nodes visible; do not silently convert them into complete
  score evidence.

**Plans:** TBD

### Phase 28: Live rocprofv3 Timing Integration

**Status:** Pending

**Goal:** Collect live source-specific ROCm profiler timing evidence during
benchmark or dataset execution.

**Requirements:** PROF-01, PROF-02, PROF-03, PROF-04

**Success criteria:**

1. Maintainers can invoke benchmark or dataset execution through `rocprofv3`
   when timing policy selects profiler-backed timing.
2. Timing evidence records source type, backend, activity domain, aggregation
   rule, tool version, GPU architecture, parsed rows, and fallback reason.
3. HIP native, Triton, PyTorch, mixed, and unknown source timing semantics are
   preserved as source-specific chimneys when needed.
4. Compile, autotune, warmup, unrelated kernel rows, and event fallback are
   excluded or explicitly labeled in timing evidence.

**Implementation notes:**

- Treat timing accuracy as the highest rule.
- Use controlled profiler output directories and parser fixtures before relying
  on local hardware traces.
- Keep PyTorch operator attribution distinct from HIP/Triton kernel activity.

**Plans:** TBD

### Phase 29: Derived AMD Scoring Workflow

**Status:** Pending

**Goal:** Connect trace JSONL, live timing evidence, AMD SOL bounds, and
baseline inputs into workload and suite reports.

**Requirements:** SCORE-01, SCORE-02, SCORE-03, SCORE-04

**Success criteria:**

1. Maintainers can generate AMD-native workload score reports from canonical
   trace JSONL, live timing evidence, AMD SOL bounds, and baseline latencies.
2. Dataset runner or additive CLI workflow can emit suite-level AMD-native
   score reports.
3. Score reports preserve evidence references for trace, timing, SOL-bound,
   baseline, and hardware-model inputs.
4. Missing timing, missing baseline, missing bound, unsupported operators, and
   unvalidated hardware appear as guarded or unscored report states.

**Implementation notes:**

- Keep score output as derived artifacts unless an additive documented output
  path is explicitly selected.
- Reuse v1.5 score guardrail models where practical.
- Do not change the existing SOL score formula or canonical trace objects.

**Plans:** TBD

### Phase 30: Compatibility and Claim Guardrails

**Status:** Pending

**Goal:** Prove public contracts remain compatible and keep CDNA3/NVIDIA
equivalence claims out of v1.6.

**Requirements:** COMPAT-01, COMPAT-02, COMPAT-03, COMPAT-04, CLAIM-01,
CLAIM-02, CLAIM-03

**Success criteria:**

1. Contract tests prove canonical trace JSONL fields and parsing behavior did
   not change.
2. Public schemas for definitions, workloads, solutions, and traces remain
   backward compatible.
3. Primary `sol-execbench` CLI defaults continue to behave as before; new
   timing or scoring outputs are opt-in, additive, or separate artifacts.
4. Documentation and score/report warnings state that real CDNA3 `gfx94*`
   full-suite validation is outside v1.6.
5. AMD-native reports do not claim NVIDIA B200, upstream SOLAR, or leaderboard
   equivalence.
6. Hardware model entries retain source, confidence, and validation status in
   every derived bound or score artifact.

**Implementation notes:**

- Run existing public-contract guardrails plus focused tests added by this
  milestone.
- Keep CDNA3 warnings active until a future hardware-validation milestone
  records real `gfx94*` evidence.
- Use this phase as the compatibility gate before milestone closure.

**Plans:** TBD

## Completed Milestones

- [x] **v1.5 AMD-native SOL Scoring and ROCm Profiler Timing** - shipped 2026-05-22. See `.planning/milestones/v1.5-ROADMAP.md`.
- [x] **v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness** - shipped 2026-05-22. See `.planning/milestones/v1.4-ROADMAP.md`.
- [x] **v1.3 Non-CDNA Issue Closure** - shipped 2026-05-22. See `.planning/milestones/v1.3-ROADMAP.md`.
- [x] **v1.2 Engineering Practice Harvest and Compatibility Guardrails** - shipped 2026-05-22. See `.planning/milestones/v1.2-ROADMAP.md`.
- [x] **v1.1 CDNA 3 Support and Migration Closure** - shipped 2026-05-21. See `.planning/milestones/v1.1-ROADMAP.md`.
- [x] **v1.0 ROCm Port** - shipped 2026-05-21. See `.planning/milestones/v1.0-ROADMAP.md`.

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SOLCOV-01 | Phase 27 | Pending |
| SOLCOV-02 | Phase 27 | Pending |
| SOLCOV-03 | Phase 27 | Pending |
| SOLCOV-04 | Phase 27 | Pending |
| PROF-01 | Phase 28 | Pending |
| PROF-02 | Phase 28 | Pending |
| PROF-03 | Phase 28 | Pending |
| PROF-04 | Phase 28 | Pending |
| SCORE-01 | Phase 29 | Pending |
| SCORE-02 | Phase 29 | Pending |
| SCORE-03 | Phase 29 | Pending |
| SCORE-04 | Phase 29 | Pending |
| COMPAT-01 | Phase 30 | Pending |
| COMPAT-02 | Phase 30 | Pending |
| COMPAT-03 | Phase 30 | Pending |
| COMPAT-04 | Phase 30 | Pending |
| CLAIM-01 | Phase 30 | Pending |
| CLAIM-02 | Phase 30 | Pending |
| CLAIM-03 | Phase 30 | Pending |
