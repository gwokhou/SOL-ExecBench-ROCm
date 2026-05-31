# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-05-31
**Milestone:** v1.20 Cross-Report Consistency and Evaluation Stability
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1 Requirements

### Cross-Report Consistency

- [x] **CONS-01**: Researcher can run a CPU-safe consistency check that loads execution closure, paper denominator, ROCm Compatibility Matrix, runtime evidence, static evidence, AMD score, AMD SOL/SOLAR, and AMD bound sanity refs without mutating canonical trace, score, timing, or public schema artifacts.
- [x] **CONS-02**: Consistency output detects contradictory status combinations across reports, including attempted/blocked denominator drift, Matrix runtime-unavailable versus attempted evidence, missing-derived-evidence versus scored reports, and stale or mismatched source refs/checksums.
- [x] **CONS-03**: Consistency findings use stable severity and reason-code vocabulary for blockers, warnings, informational notes, and claim-boundary violations.
- [x] **CONS-04**: Consistency reports are deterministic JSON plus Markdown summaries with bounded relative refs, checksums, and no embedded raw logs, proprietary kernels, credentials, or absolute temp paths.
- [x] **CONS-05**: Consistency tooling remains diagnostic-only and cannot upgrade Docker, runtime, static, denominator, or AMD SOL/SOLAR evidence into native-host validation, score authority, paper parity, or leaderboard authority.

### Evaluation Stability

- [x] **STAB-01**: Project defines a strict `evaluation_stability.v1` sidecar model recording timing backend, warmup count, measured repeat count, runtime distribution, selected summary statistic, clock policy, synchronization policy, and source trace refs.
- [x] **STAB-02**: Stability classification distinguishes stable, noisy, insufficient-samples, missing-timing, clock-unlocked, profiler-overhead-risk, and backend-unsupported states with stable reason codes.
- [x] **STAB-03**: Stability reports compute deterministic variance/dispersion summaries from existing timing evidence without changing canonical trace JSONL, correctness, timing, score, or evaluator semantics.
- [x] **STAB-04**: ROCm E2E coverage demonstrates a real benchmark path can emit or validate stability evidence for representative HIP/C++ or PyTorch ROCm workloads.
- [x] **STAB-05**: Stability documentation explains that timing quality supports interpretation only and does not itself create correctness, score, paper-parity, native-host, or leaderboard authority.

### Claim Upgrade Rules

- [x] **CLAIM-01**: Project defines machine-readable claim-upgrade rules for diagnostic-only, container-validated, native-host-validated, score-authoritative, paper-parity-candidate, and leaderboard-ready states.
- [x] **CLAIM-02**: Claim-upgrade evaluation requires explicit supporting evidence refs and rejects upgrades when required closure, denominator, Matrix, runtime, stability, AMD score, AMD SOL/SOLAR, or hardware validation evidence is missing or contradictory.
- [x] **CLAIM-03**: Claim-upgrade output explains unmet prerequisites and next-evidence hints without changing report authority fields unless a future milestone explicitly implements the required validation evidence.
- [x] **CLAIM-04**: Existing v1.19 and earlier diagnostic artifacts remain authority-false by default unless claim-upgrade rules prove every required prerequisite.

### Trust Summary

- [ ] **TRUST-01**: Researcher can generate a deterministic trust summary that combines consistency, stability, claim-upgrade, closure, denominator, Matrix, AMD score, and AMD SOL/SOLAR status into a concise JSON and Markdown review artifact.
- [ ] **TRUST-02**: Trust summary clearly separates internally consistent, stable enough to interpret, evidence missing, diagnostic-only, and claim upgrade blocked outcomes.
- [ ] **TRUST-03**: Trust summary references source reports by bounded refs and checksums rather than duplicating full payloads.
- [ ] **TRUST-04**: Trust summary gives actionable next steps for future CDNA3/MI300X/native-host/paper-scale validation without claiming that v1.20 performed those validations.

### Documentation And Guardrails

- [ ] **DOCS-01**: Documentation explains how to generate and interpret consistency lint, evaluation stability, claim-upgrade, and trust summary artifacts.
- [ ] **DOCS-02**: Documentation states that v1.20 does not add full 235-problem paper validation, CDNA3/MI300X/CDNA4 validation, native-host Matrix authority, hosted leaderboard readiness, or upstream SOLAR parity.
- [ ] **DOCS-03**: CPU-safe tests cover consistency contradictions, stability classification, claim-upgrade rejection, trust summary rendering, deterministic serialization, and docs claim-boundary wording.
- [ ] **DOCS-04**: Public examples or fixtures show representative consistent, contradictory, noisy, and claim-blocked report shapes with bounded refs, checksums, and diagnostic-only wording.
- [ ] **DOCS-05**: Existing public contracts remain stable: canonical Trace, Definition, Workload, Solution, correctness, timing, score, and evaluator contract semantics are unchanged by v1.20 evidence features.

## Future Requirements

### Hardware Validation

- **HW-01**: CDNA3/MI300X or CDNA4 live hardware validation can use v1.20 consistency and stability artifacts as preflight and post-run review inputs.

### Paper Parity

- **PAPER-01**: Full 235-problem paper validation can require passing consistency, stability, and claim-upgrade checks before any paper-parity candidate claim.

### Leaderboard Readiness

- **LEADER-01**: Hosted leaderboard or remote submission service can require claim-upgrade and trust-summary gates after sandboxing, anti-cheat, baseline, and policy work exists.

### Profiling Diagnostics

- **PROF-01**: ROCm profiler/Instruction Roofline diagnostics can feed stability and trust summaries once counter availability and profiling-overhead uncertainty are modeled.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full 235-problem real-hardware validation | v1.20 audits evidence quality and consistency before expensive paper-scale runs. |
| CDNA 3, MI300X, CDNA 4, or native-host ROCm validation expansion | Deferred to a later hardware milestone after consistency and stability gates exist. |
| Hosted leaderboard or remote submission service | Requires separate submission isolation, anti-cheat, hardware authority, policy, and baseline work. |
| Upstream SOLAR parity claim | v1.20 can check local evidence consistency but does not validate upstream SOLAR equivalence. |
| Making stability evidence score authority | Stability supports interpretation; it does not change score semantics or eligibility. |
| Canonical trace, scoring, timing, correctness, or evaluator contract changes | v1.20 evidence remains sidecar-only and diagnostic. |
| New databases, dashboards, or remote services | Local JSON/Markdown sidecars and scripts are sufficient for this milestone. |
| Dependency relocking or Docker privilege expansion | Not needed for consistency and stability diagnostics. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CONS-01 | Phase 89 | Complete |
| CONS-02 | Phase 89 | Complete |
| CONS-03 | Phase 89 | Complete |
| CONS-04 | Phase 89 | Complete |
| CONS-05 | Phase 89 | Complete |
| STAB-01 | Phase 90 | Complete |
| STAB-02 | Phase 90 | Complete |
| STAB-03 | Phase 90 | Complete |
| STAB-04 | Phase 90 | Complete |
| STAB-05 | Phase 90 | Complete |
| CLAIM-01 | Phase 91 | Complete |
| CLAIM-02 | Phase 91 | Complete |
| CLAIM-03 | Phase 91 | Complete |
| CLAIM-04 | Phase 91 | Complete |
| TRUST-01 | Phase 92 | Pending |
| TRUST-02 | Phase 92 | Pending |
| TRUST-03 | Phase 92 | Pending |
| TRUST-04 | Phase 92 | Pending |
| DOCS-01 | Phase 93 | Pending |
| DOCS-02 | Phase 93 | Pending |
| DOCS-03 | Phase 93 | Pending |
| DOCS-04 | Phase 93 | Pending |
| DOCS-05 | Phase 93 | Pending |

**Coverage:**
- v1 requirements: 23 total, 14 complete
- Mapped to phases: 23
- Unmapped: 0

---
*Requirements defined: 2026-05-31*
*Last updated: 2026-05-31 after Phase 91 completion*
