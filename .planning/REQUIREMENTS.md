# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-05-22
**Milestone:** v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## v1.6 Requirements

### AMD SOLAR Coverage

- [x] **SOLCOV-01**: Maintainer can run an AMD SOL analyzer with broader operator coverage than the v1.5 matmul and broad elementwise foundation.
- [x] **SOLCOV-02**: Maintainer can inspect per-operation coverage status as supported, inexact, or unsupported with rationale.
- [x] **SOLCOV-03**: Maintainer can produce a derived coverage summary that reports supported, inexact, and unsupported operation counts before score generation.
- [x] **SOLCOV-04**: AMD SOL bound artifacts preserve per-op FLOP, byte, limiting-resource, confidence, and hardware-model evidence without mutating canonical trace JSONL.

### Live Profiler Timing

- [ ] **PROF-01**: Maintainer can collect live `rocprofv3` timing evidence during benchmark or dataset execution when the timing policy selects profiler-backed timing.
- [ ] **PROF-02**: Timing evidence records source type, timer backend, activity domain, aggregation rule, tool version, GPU architecture, parsed rows, and fallback reason when applicable.
- [ ] **PROF-03**: HIP native, Triton, PyTorch, mixed, and unknown sources preserve source-specific timing semantics instead of forcing one unified timer口径.
- [ ] **PROF-04**: Live timing excludes or explicitly labels compile, autotune, warmup, unrelated kernel rows, and event-timing fallback so measured latency remains auditable.

### Derived Scoring Workflow

- [ ] **SCORE-01**: Maintainer can generate AMD-native workload score reports from canonical trace JSONL, live timing evidence, AMD SOL bound artifacts, and baseline latency inputs.
- [ ] **SCORE-02**: Maintainer can generate suite-level AMD-native score reports through the dataset runner or an additive CLI workflow.
- [ ] **SCORE-03**: Score reports preserve evidence references for trace, timing, SOL-bound, baseline, and hardware-model inputs.
- [ ] **SCORE-04**: Score reports explicitly mark missing timing, missing baseline, missing bound, unsupported operators, or unvalidated hardware as guarded or unscored states.

### Compatibility Guardrails

- [ ] **COMPAT-01**: Existing canonical trace JSONL fields and parsing behavior remain unchanged.
- [ ] **COMPAT-02**: Existing public schemas remain backward compatible for definitions, workloads, solutions, and traces.
- [ ] **COMPAT-03**: Existing primary `sol-execbench` CLI defaults remain compatible; new timing or scoring outputs are opt-in, additive, or separate derived artifacts.
- [ ] **COMPAT-04**: Contract tests verify trace/schema/primary CLI compatibility after the analyzer, timing, and scoring workflow changes.

### Claim Boundaries

- [ ] **CLAIM-01**: v1.6 documentation and report warnings state that real CDNA3 `gfx94*` full-suite validation is not part of this milestone.
- [ ] **CLAIM-02**: AMD-native score reports do not claim NVIDIA B200, upstream SOLAR, or leaderboard equivalence.
- [ ] **CLAIM-03**: Hardware model entries retain source, confidence, and validation status in every derived bound or score artifact.

## Future Requirements

### Hardware Validation

- **HWVAL-01**: Maintainer can run and record a real CDNA3 `gfx94*` full adapted suite validation.
- **HWVAL-02**: Maintainer can promote AMD hardware model entries from provisional or unvalidated to validated after architecture-specific evidence is recorded.

### Upstream SOLAR Parity

- **SOLPAR-01**: Maintainer can compare AMD-native analyzer coverage against upstream SOLAR categories beyond the v1.6 supported subset.
- **SOLPAR-02**: Maintainer can model additional architecture-specific dtype and execution paths once authoritative AMD peak inputs and validation evidence exist.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real CDNA3 `gfx94*` full-suite validation | User explicitly kept CDNA3 validation outside this milestone. |
| NVIDIA B200, upstream SOLAR, or leaderboard equivalence claims | v1.6 produces AMD-native derived artifacts, not cross-vendor leaderboard validation. |
| Mutating canonical trace JSONL for timing, SOL, or score evidence | The milestone hard constraint requires existing trace compatibility. |
| Breaking or replacing primary `sol-execbench` CLI defaults | New workflow outputs must be additive or opt-in. |
| Forcing one timer口径 across HIP native, Triton, and PyTorch | Timing accuracy is the highest rule; source-specific timing chimneys are required when semantics differ. |

## Traceability

Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SOLCOV-01 | Phase 27 | Complete |
| SOLCOV-02 | Phase 27 | Complete |
| SOLCOV-03 | Phase 27 | Complete |
| SOLCOV-04 | Phase 27 | Complete |
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

**Coverage:**
- v1.6 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0

---
*Requirements defined: 2026-05-22*
*Last updated: 2026-05-22 after Phase 27 completion*
