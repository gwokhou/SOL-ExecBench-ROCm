# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-06-15
**Milestone:** v1.36 SOL Agent Feedback Sidecar Producer
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1.36 Requirements

### Contract and Capability

- [x] **CNTR-01**: Downstream maintainers can inspect documented optional
  `agent_feedback.sidecar.v1` and `profile_summary.sidecar.v1` capabilities in
  the evaluator contract without changing required trace field groups or
  contract compatibility for older consumers.
- [x] **CNTR-02**: Documentation defines the SOL agent-feedback/profile-summary
  sidecar as diagnostic-only and explicitly not correctness, timing, score,
  evidence-tier, confirmed-improvement, release-gate, cutover, paper-parity, or
  leaderboard authority.
- [x] **CNTR-03**: Contract tests prove optional feedback capabilities do not
  add fields to canonical Trace JSONL correctness, timing, scoring, or status
  semantics.

### Sidecar Schema and Generation

- [x] **SIDE-01**: Maintainer can validate a strict
  `sol_execbench.agent_feedback.v1` sidecar schema with bounded status,
  reason-code, bottleneck, recommendation, limitation, authority, and citation
  fields.
- [x] **SIDE-02**: Evaluation runs with a trace output path can persist
  `trace.jsonl.agent-feedback.json` beside the canonical trace without changing
  trace emission, evaluation status, or existing profile/static/environment
  sidecar behavior.
- [x] **SIDE-03**: Sidecar generation summarizes existing trace/profile/static
  evidence/evaluation-stability inputs into prompt-safe bottleneck,
  recommendation, and limitation entries without embedding raw trace rows, raw
  profiler dumps, full source, prompt text, or unstable absolute temporary paths.
- [x] **SIDE-04**: Missing, skipped, unavailable, partial, and failed optional
  diagnostic inputs produce explicit sidecar availability and limitation states
  instead of failing benchmark execution.

### Freshness and Artifact Identity

- [ ] **IDEN-01**: Sidecars include freshness identity covering trace path,
  generated timestamp, SOL contract version, target/run identity when available,
  candidate/source hash when available, and referenced artifact checksums.
- [ ] **IDEN-02**: Artifact citations use compact path/checksum references for
  profile, static evidence, environment, stability, and trace-adjacent inputs
  without leaking raw profiler directories into prompt-facing summaries.
- [ ] **IDEN-03**: Validator helpers classify stale or identity-mismatched
  sidecars as diagnostic invalid/stale states while leaving canonical trace
  validity unchanged.

### Claim Governance

- [ ] **GOVR-01**: Sidecar schema enforces `diagnostic_only=true` and false
  authority flags for correctness, timing, performance, score, evidence tier,
  confirmed improvement, release gate, cutover, paper parity, and leaderboard
  readiness.
- [ ] **GOVR-02**: Guardrail tests reject contradictory-authority payloads and
  prove sidecar presence, absence, parse failure, or stale identity cannot
  promote score authority, evidence tier, claim-upgrade status, release gates,
  or cutover eligibility.
- [ ] **GOVR-03**: Existing public claim-boundary docs and evidence-quality docs
  describe the feedback sidecar as next-experiment guidance only.

### HIP Integration Fixtures

- [ ] **FIXT-01**: Repository fixtures include valid, missing/unavailable,
  malformed, stale, partial, and contradictory-authority feedback sidecar cases
  for HIP adapter and runtime tests.
- [ ] **FIXT-02**: Fixture docs explain how HIP should map SOL bottleneck,
  recommendation, limitation, and citation fields into closed prompt-safe
  consumer taxonomies, with unknown values downgraded safely.
- [ ] **FIXT-03**: CPU-safe tests verify generated fixtures and example sidecars
  remain deterministic and contain no raw profiler dump content, full source,
  raw trace rows, or absolute temporary paths.

## Future Requirements

- **SIDE-F01**: SOL parses profiler counter artifacts into a stable hardware
  counter taxonomy for occupancy, VGPR/SGPR, LDS, scratch, bandwidth, cache, and
  utilization.
- **SIDE-F02**: SOL emits confidence-scored bottleneck classification from
  multiple corroborated profile sources.
- **SIDE-F03**: Feedback participates in candidate ranking after a separate
  evidence-model and governance milestone.
- **SIDE-F04**: Cross-run feedback accumulation summarizes repeated bottlenecks
  across campaigns.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Changing canonical Trace JSONL schema | HIP v1.26 expects feedback as optional sidecars while trace remains the benchmark truth contract. |
| Treating feedback as score, timing, evidence-tier, release-gate, or cutover authority | The sidecar is diagnostic next-experiment guidance only. |
| Requiring profiler/static evidence for every run | Optional tool availability varies and benchmark execution must remain useful without diagnostics. |
| Raw profiler dump or full trace prompt export | Prompt-facing feedback must stay bounded, cited, and normalized. |
| HIP adapter/runtime implementation | HIP Playground owns `hip_agent.execbench`, `ProfileDigest`, strategy hints, and runtime prompt assembly. |
| Paper-scale validation, MI300X/CDNA3 full-suite validation, CDNA4 validation, hosted leaderboard, or hard sandboxing | These remain deferred project boundaries unrelated to the feedback sidecar producer. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CNTR-01 | Phase 181 | Complete |
| CNTR-02 | Phase 181 | Complete |
| CNTR-03 | Phase 181 | Complete |
| SIDE-01 | Phase 182 | Complete |
| SIDE-02 | Phase 182 | Complete |
| SIDE-03 | Phase 182 | Complete |
| SIDE-04 | Phase 182 | Complete |
| IDEN-01 | Phase 183 | Pending |
| IDEN-02 | Phase 183 | Pending |
| IDEN-03 | Phase 183 | Pending |
| GOVR-01 | Phase 184 | Pending |
| GOVR-02 | Phase 184 | Pending |
| GOVR-03 | Phase 184 | Pending |
| FIXT-01 | Phase 185 | Pending |
| FIXT-02 | Phase 185 | Pending |
| FIXT-03 | Phase 185 | Pending |

**Coverage:**
- v1.36 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0

---
*Requirements defined: 2026-06-15*
*Last updated: 2026-06-15 after roadmap traceability mapping*
