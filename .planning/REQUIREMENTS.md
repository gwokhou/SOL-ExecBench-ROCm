# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-05-22
**Milestone:** v1.5 AMD-native SOL Scoring and ROCm Profiler Timing
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## v1.5 Requirements

### Timing Semantics

- [x] **TIME-01**: Maintainer can classify measured work as HIP native, Triton, PyTorch, or mixed source type before selecting a timing backend.
- [x] **TIME-02**: Maintainer can inspect a `source_type -> timer_backend -> interpretation` timing policy table.
- [x] **TIME-03**: When a unified timing口径 would reduce accuracy, the benchmark exposes source-specific timing chimneys instead of forcing one interpretation.
- [x] **TIME-04**: Timing documentation states whether each timer measures kernel activity, HIP runtime/API activity, PyTorch operator attribution, or fallback event timing.

### ROCm Profiler Timing

- [x] **PROF-01**: Maintainer can collect and parse `rocprofv3` profiler timing evidence for representative benchmark executions.
- [x] **PROF-02**: The default timing path uses profiler-backed timing when it is the most accurate supported backend for the source type.
- [x] **PROF-03**: Any fallback timing path is labeled with backend, reason, and interpretation in timing evidence.
- [x] **PROF-04**: Timing evidence includes tool version, GPU architecture, activity domain, aggregation rule, and parsed timing rows needed to audit the measured duration.

### AMD SOL Bound

- [x] **SOL-01**: Maintainer can run a SOLAR-like graph extraction foundation for supported benchmark workloads.
- [x] **SOL-02**: FLOP and byte analysis records supported, inexact, and unsupported operations with confidence and rationale.
- [x] **SOL-03**: AMD hardware model inputs record architecture, dtype or execution path, peak-value source, confidence, and validation status.
- [x] **SOL-04**: Per-op and aggregate AMD SOL bound artifacts are generated before AMD-native scoring is allowed.

### AMD-native Scoring

- [x] **SCORE-01**: Maintainer can generate per-problem AMD-native scores from measured timing and AMD SOL bound artifacts.
- [x] **SCORE-02**: Baseline ingestion and comparison are supported without claiming NVIDIA B200, SOLAR, or leaderboard equivalence.
- [x] **SCORE-03**: Suite-level aggregation preserves references to each workload's timing and SOL-bound evidence.
- [x] **SCORE-04**: Score and report outputs include claim guardrails for unsupported, incomplete, or unvalidated evidence.

### Compatibility and Claims

- [x] **COMPAT-01**: v1.5 does not regress existing CLI behavior, solution schema, canonical trace JSONL, eval-driver correctness semantics, or reward-hack defenses.
- [x] **COMPAT-02**: SOL and timing evidence are emitted as derived artifacts unless an additive documented output is explicitly introduced.
- [x] **CLAIM-01**: v1.5 does not include real CDNA3 `gfx94*` full-suite validation.
- [x] **CLAIM-02**: CDNA3 hardware model or readiness scaffolding is labeled as unvalidated and does not produce CDNA3 hardware-validation claims.

## Future Requirements

### CDNA3 Hardware Validation

- **CDNA-HW-01**: Adapted test suite passes on at least one real CDNA3 GPU environment and full evidence is recorded.
- **CDNA-HW-02**: Documentation can claim CDNA3 hardware validation only after recorded `gfx94*` full-suite pass.

### Broader SOL Coverage

- **SOL-COV-01**: Additional operator analyzers extend SOL-bound coverage beyond the v1.5 foundation.
- **SOL-COV-02**: Hardware model entries can be promoted from provisional to validated after architecture-specific evidence is recorded.

### Leaderboard Equivalence

- **LEADER-01**: Any NVIDIA B200, SOLAR, or upstream leaderboard equivalence claim requires a separate validation and methodology milestone.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real CDNA3 `gfx94*` full-suite validation | User explicitly excluded CDNA3 validation from this milestone. |
| Claiming NVIDIA B200/SOLAR leaderboard equivalence | v1.5 creates AMD-native interpretation, not cross-vendor leaderboard validation. |
| Forcing one timing口径 for HIP, Triton, and PyTorch | Timing accuracy is the highest rule; source-specific chimneys are allowed when needed. |
| Mutating canonical trace JSONL for SOL/timing evidence | Derived artifacts preserve existing public benchmark contracts. |
| Replacing eval-driver correctness or reward-hack semantics | The original paper's benchmark semantics remain the baseline. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

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
| SCORE-01 | Phase 26 | Complete |
| SCORE-02 | Phase 26 | Complete |
| SCORE-03 | Phase 26 | Complete |
| SCORE-04 | Phase 26 | Complete |
| COMPAT-01 | Phase 26 | Complete |
| COMPAT-02 | Phase 26 | Complete |
| CLAIM-01 | Phase 26 | Complete |
| CLAIM-02 | Phase 26 | Complete |

**Coverage:**
- v1.5 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0

---
*Requirements defined: 2026-05-22*
*Last updated: 2026-05-22 after Phase 26 completion*
