# Requirements: SOL ExecBench ROCm Port v1.31

**Defined:** 2026-06-08
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on
AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1.31 Requirements

### RDNA4 Clock And Timing Closure

- [x] **RDNA4-FU-TIME-01**: Host setup provides an auditable way to install and
  verify passwordless `rocm-smi` sudoers coverage for RDNA4 clock lock and
  reset commands.
- [x] **RDNA4-FU-TIME-02**: RDNA4 clock-lock evidence is rerun after sudoers
  coverage is fixed, with failures captured as explicit blockers.
- [x] **RDNA4-FU-TIME-03**: RDNA4 timing collection records whether
  profiler-backed `rocprofv3` kernel activity timing is available for the
  selected workloads, without overclaiming fallback event timing.
- [x] **RDNA4-FU-TIME-04**: The milestone identifies why v1.31 timing sidecars
  remain PyTorch/device-event fallback and distinguishes source-language policy
  routing from `rocprofv3` tool availability or parser failure.

### RDNA4 Derived Sidecar Memory Closure

- [x] **RDNA4-FU-DERIVED-01**: The 56 temporary derived sidecar exclusions are
  retried through isolated memory-capped jobs or classified with per-problem
  memory blockers.
- [x] **RDNA4-FU-DERIVED-02**: Derived sidecar retry results update local
  evidence reports without taking down Codex or the calling shell.

### RDNA4 Missing Trace And Failure Triage

- [x] **RDNA4-FU-TRACE-01**: The 12 `missing_trace` workload records are
  reproduced or classified by root cause.
- [x] **RDNA4-FU-FAIL-01**: The 146 failed RDNA4 workloads are grouped by
  failure class, representative examples, likely root cause, and next action.
- [x] **RDNA4-FU-CLAIM-01**: Public and planning docs continue to block stronger
  RDNA4 claims until timing, missing traces, sidecar exclusions, and failed
  workloads have accepted closure.
- [x] **RDNA4-FU-CLOSE-01**: Milestone closure records concrete Phase 147
  memory optimization and targeted retry attempts before accepting residual
  boundaries, next-step priorities, and no-claim-upgrade constraints.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Historical open-artifact cleanup | Explicitly excluded by the user for this follow-up. |
| Hosted leaderboard authority | Operational leaderboard support remains outside this milestone. |
| NVIDIA B200 or upstream SOLAR equivalence | RDNA4 follow-up does not validate NVIDIA hardware or upstream SOLAR parity. |
| CDNA3/MI300X or CDNA4 validation | This milestone is RDNA4 follow-up only. |
| CDNA3-family, CDNA4, or native-host ROCm validation expansion | Deferred validation surfaces remain outside this RDNA4 follow-up. |
| Actual MI300X full-suite execution under CDNA3 in v1.28 | Remains historical deferred scope and is not reopened by v1.31. |
| MI300X and MI308X are sibling GPU products | Product distinction remains historical context and is not reopened by v1.31. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| RDNA4-FU-TIME-01 | Phase 142 | Complete |
| RDNA4-FU-TIME-02 | Phase 143 | Complete |
| RDNA4-FU-TIME-03 | Phase 143 | Complete |
| RDNA4-FU-TIME-04 | Phase 147 | Complete |
| RDNA4-FU-DERIVED-01 | Phase 144 | Complete |
| RDNA4-FU-DERIVED-02 | Phase 144 | Complete |
| RDNA4-FU-TRACE-01 | Phase 145 | Complete |
| RDNA4-FU-FAIL-01 | Phase 146 | Complete |
| RDNA4-FU-CLAIM-01 | Phase 146 | Complete |
| RDNA4-FU-CLOSE-01 | Phase 147 | Complete |

**Coverage:**
- v1.31 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0
