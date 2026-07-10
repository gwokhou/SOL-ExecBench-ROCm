# Roadmap: SOL ExecBench ROCm Port

## Milestones

- **v1.38 Upgrade SOL Evidence Contract for Confirmed Benchmark Claims** - Phases 190-194 (active)
- **v1.37 Profile Summary Sidecar v1** - Phases 186-189 (shipped 2026-06-16)
  See `.planning/milestones/v1.37-ROADMAP.md`.
- **v1.36 SOL Agent Feedback Sidecar Producer** - Phases 181-185 (shipped 2026-06-16)
  See `.planning/milestones/v1.36-ROADMAP.md`.
- **v1.35 Script Parallelism and Safety Hardening** - Phases 175-180 (shipped 2026-06-11)
  See `.planning/milestones/v1.35-ROADMAP.md`.
- **v1.34 RDNA4 Readiness Blocker Closure** - Phases 170-174 (shipped 2026-06-09)
  See `.planning/milestones/v1.34-ROADMAP.md`.
- Complete **v1.33 RDNA4 Benchmark-Grade Evidence Closure** - Phases 163-169
  See `.planning/milestones/v1.33-ROADMAP.md`.
- Complete **v1.32 RDNA4 Profiler Timing Coverage Closure** - Phases 148-162
  See `.planning/milestones/v1.32-ROADMAP.md`.
- Complete **v1.31 RDNA4 Follow-Up Evidence Hardening** - Phases 142-147
  See `.planning/milestones/v1.31-ROADMAP.md`.
- Complete **v1.30 RDNA4 Benchmark-Grade Validation Closure** - Phases 136-141
  See `.planning/milestones/v1.30-ROADMAP.md`.
- Complete **v1.29 Dataset Migration and Compliance** - Phases 131-135
  See `.planning/milestones/v1.29-ROADMAP.md`.
- Earlier milestones are archived under `.planning/milestones/`.

## Current Position

**Status:** v1.38 executing. Phases 190-193 are verified; phase 194 remains
pending. Phase 193 shipped 2026-07-10 (measured baseline provenance + five-state
coverage validation + official score gate wiring). See
`.planning/phases/193-measured-baseline-provenance-and-coverage/193-VERIFICATION.md`
for the 193 verification.

## v1.38 Upgrade SOL Evidence Contract for Confirmed Benchmark Claims

**Goal:** Enable HIP Playground to make confirmed benchmark pass/fail decisions
from SOL-provided profiling, official score, and measured baseline evidence.

| Phase | Name | Goal | Requirements | Success Criteria |
|-------|------|------|--------------|------------------|
| 190 | Profiler Artifact Registration Closure | Make requested successful `rocprofv3` profile runs produce discoverable artifacts and citations. | PROF-01, PROF-02, PROF-03 | Verified 2026-06-21 |
| 191 | Structured Profile Summary Evidence | Expand profile summary into structured profiling evidence with bottleneck hints while preserving diagnostic authority boundaries. | PSUM-01, PSUM-02, PSUM-03 | Verified 2026-06-21 |
| 192 | Official Score Evidence Contract | Add official benchmark score evidence with score source, aggregation policy, and valid-run non-null score semantics. | SCOR-01, SCOR-02, SCOR-03 | Verified 2026-06-21 |
| 193 | Measured Baseline Provenance and Coverage | Add measured baseline evidence and coverage validation for confirmed benchmark claims. | BASE-01, BASE-02, BASE-03 | 4 |
| 194 | HIP Confirmed Evidence Package | Publish contract capabilities, fixtures, docs, and guardrails for HIP cutover-gate decisions. | GATE-01, GATE-02, GATE-03 | 4 |

### Phase 190: Profiler Artifact Registration Closure

Goal: Make requested successful `rocprofv3` profile runs produce discoverable
artifacts and citations.

Requirements: PROF-01, PROF-02, PROF-03

Success criteria:
1. Real or fixture-backed `rocprofv3` profile output files are discovered across
   supported output formats and nested/version-specific layouts.
2. Successful profiler runs with files no longer produce
   `rocprof_no_registered_artifacts` or equivalent missing-artifact blockers.
3. Profile result metadata records artifact kind, compact path, size, checksum
   where practical, status, and failure/unavailable/partial reason codes.
4. Tests cover success, no-files, command failure, unavailable profiler, and
   nested output layout cases.

### Phase 191: Structured Profile Summary Evidence

Goal: Expand profile summary into structured profiling evidence with bottleneck
hints while preserving diagnostic authority boundaries.

Requirements: PSUM-01, PSUM-02, PSUM-03

Success criteria:
1. Profile summary includes workload/kernel metric records derived from bounded
   profiler artifacts and trace/profile metadata.
2. Bottleneck hints use stable AMD-oriented categories for compute, memory/L2,
   LDS, launch/dispatch, and insufficient-counter states.
3. Profile summary cites concrete trace, profile metadata, and profiler
   artifacts used to build the summary.
4. Governance tests prove profile summary and agent feedback sidecars remain
   diagnostic-only and cannot satisfy score, release-gate, or cutover authority.

### Phase 192: Official Score Evidence Contract

Goal: Add official benchmark score evidence with score source, aggregation
policy, and valid-run non-null score semantics.

Requirements: SCOR-01, SCOR-02, SCOR-03

Success criteria:
1. SOL emits official score evidence with schema version, score source,
   aggregation policy, score value, scored/unscored counts, and input refs.
2. Valid runs with measured latency, official measured baseline, and SOL bound
   evidence produce non-null official score evidence.
3. Reports distinguish official score from diagnostic speedup and provisional
   AMD-native derived score output.
4. Placeholder/reference baseline fallback is blocked or explicitly classified
   as non-confirmed for official score claims.

### Phase 193: Measured Baseline Provenance and Coverage

Goal: Add measured baseline evidence and coverage validation for confirmed
benchmark claims.

Requirements: BASE-01, BASE-02, BASE-03

Success criteria:
1. Measured baseline evidence records trace pointer, hardware, ROCm version, SOL
   version, target identity, timing policy, workload coverage, and timestamp.
2. Coverage validation reports confirmed, missing, stale, mismatched, and
   placeholder baseline states with stable reason codes.
3. Baseline evidence integrates with official score generation without treating
   `reference_latency_ms` as release-defined baseline evidence.
4. Tests cover complete coverage, partial coverage, hardware mismatch, timing
   policy mismatch, stale trace pointer, and placeholder baseline rejection.

### Phase 194: HIP Confirmed Evidence Package

Goal: Publish contract capabilities, fixtures, docs, and guardrails for HIP
cutover-gate decisions.

Requirements: GATE-01, GATE-02, GATE-03

Success criteria:
1. Evaluator contract advertises confirmed benchmark evidence capabilities and
   source-boundary claims.
2. HIP-facing fixtures cover confirmed pass, missing score, missing baseline,
   placeholder baseline, profiler partial, and diagnostic-only sidecar cases.
3. Documentation explains which SOL artifacts HIP must consume for confirmed
   pass/fail and which artifacts remain diagnostic-only.
4. Valid fixture runs remove `missing_score`, `missing_baseline`, and
   `placeholder_baseline` blockers while invalid fixture runs keep precise
   blocker reason codes.

<details>
<summary>v1.37 Profile Summary Sidecar v1 (Phases 186-189) -- SHIPPED 2026-06-16</summary>

- [x] Phase 186: Profile Summary Contract and Schema (1/1 plans)
- [x] Phase 187: Profile Summary Producer and CLI Persistence (1/1 plans)
- [x] Phase 188: Profile Summary Freshness and Governance (1/1 plans)
- [x] Phase 189: HIP Profile Summary Fixtures and Docs (1/1 plans)

Archive: `.planning/milestones/v1.37-ROADMAP.md`

</details>

<details>
<summary>v1.36 SOL Agent Feedback Sidecar Producer (Phases 181-185) -- SHIPPED 2026-06-16</summary>

- [x] Phase 181: Feedback Contract and Capability Surface (1/1 plans)
- [x] Phase 182: Diagnostic Sidecar Schema and Generator (2/2 plans)
- [x] Phase 183: Freshness Identity and Artifact References (1/1 plans)
- [x] Phase 184: Governance Guardrails and Compatibility Fixtures (1/1 plans)
- [x] Phase 185: HIP Consumer Integration Package and Docs (1/1 plans)

Archive: `.planning/milestones/v1.36-ROADMAP.md`

</details>

<details>
<summary>v1.35 Script Parallelism and Safety Hardening (Phases 175-180) -- SHIPPED 2026-06-11</summary>

- [x] Phase 175: PID Lock Module (1/1 plans)
- [x] Phase 176: Timing Isolation Audit (1/1 plans)
- [x] Phase 177: Profiler Timing Batch Parallelism (1/1 plans)
- [x] Phase 178: Derived Script Parallelism (1/1 plans)
- [x] Phase 179: Evaluation Stability Extension and Integration Tests (1/1 plans)
- [x] Phase 180: Timing Environment Hardening and Overhead Calibration (2/2 plans)

Archive: `.planning/milestones/v1.35-ROADMAP.md`

</details>
