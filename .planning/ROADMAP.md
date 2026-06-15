# Roadmap: SOL ExecBench ROCm Port

## Milestones

- **v1.36 SOL Agent Feedback Sidecar Producer** - Phases 181-185 (planning)
- **v1.35 Script Parallelism and Safety Hardening** - Phases 175-180 (shipped 2026-06-11)
- **v1.34 RDNA4 Readiness Blocker Closure** - Phases 170-174 (shipped 2026-06-09)
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

**Status:** Planning v1.36 SOL Agent Feedback Sidecar Producer.

<details open>
<summary>v1.36 SOL Agent Feedback Sidecar Producer (Phases 181-185) -- PLANNING</summary>

**Milestone Goal:** Deliver the optional diagnostic SOL
agent-feedback/profile-summary sidecar producer that HIP Playground v1.26 needs,
while preserving canonical Trace JSONL as the only authority for correctness,
timing, scoring, and release-gate semantics.

- [ ] **Phase 181: Feedback Contract and Capability Surface** -- Add optional
  feedback/profile-summary capabilities and documentation boundaries without
  changing canonical trace semantics.
- [ ] **Phase 182: Diagnostic Sidecar Schema and Generator** -- Define strict
  `sol_execbench.agent_feedback.v1` models and persist
  `trace.jsonl.agent-feedback.json` beside trace outputs.
- [ ] **Phase 183: Freshness Identity and Artifact References** -- Attach
  trace/run/candidate identity and compact artifact citations so consumers can
  reject stale or mismatched feedback.
- [ ] **Phase 184: Governance Guardrails and Compatibility Fixtures** -- Prove
  sidecars remain diagnostic-only across valid, invalid, stale, and
  contradictory states.
- [ ] **Phase 185: HIP Consumer Integration Package and Docs** -- Provide
  fixtures, examples, and docs that HIP can use for adapter/runtime tests.

</details>

<details>
<summary>v1.35 Script Parallelism and Safety Hardening (Phases 175-180) -- SHIPPED 2026-06-11</summary>

- [x] Phase 175: PID Lock Module (1/1 plans)
- [x] Phase 176: Timing Isolation Audit (1/1 plans)
- [x] Phase 177: Profiler Timing Batch Parallelism (1/1 plans)
- [x] Phase 178: Derived Script Parallelism (1/1 plans)
- [x] Phase 179: Evaluation Stability Extension and Integration Tests (1/1 plans)
- [x] Phase 180: Timing Environment Hardening and Overhead Calibration (2/2 plans)

</details>

<details>
<summary>v1.34 RDNA4 Readiness Blocker Closure (Phases 170-174) -- SHIPPED 2026-06-09</summary>

- [x] Phase 170: Custom Input Evaluator Readiness (1/1 plans)
- [x] Phase 171: Custom Input Coverage Recompute (1/1 plans)
- [x] Phase 172: Quant Readiness Triage (1/1 plans)
- [x] Phase 173: FlashInfer Readiness Split (1/1 plans)
- [x] Phase 174: RDNA4 Readiness Closure Report and Claim Guardrails (1/1 plans)

</details>

## Phase Details

### Phase 181: Feedback Contract and Capability Surface

**Goal:** Downstream consumers can discover optional SOL feedback sidecar support
and read the authority boundary before implementation.
**Depends on:** Phase 180
**Requirements:** CNTR-01, CNTR-02, CNTR-03

**Success Criteria:**
1. `sol-execbench contract --json` advertises optional agent-feedback and
   profile-summary capability tokens without adding required trace fields.
2. Documentation says feedback/profile-summary sidecars are diagnostic-only and
   not correctness, timing, score, evidence-tier, release-gate, cutover,
   paper-parity, or leaderboard authority.
3. Contract tests prove canonical Trace JSONL field groups and status semantics
   remain unchanged.

**Plans:** 1 plan

- [ ] 181-01-PLAN.md -- Optional feedback capability tokens, contract tests,
  and evaluator-contract documentation.

### Phase 182: Diagnostic Sidecar Schema and Generator

**Goal:** SOL can generate a strict optional sidecar that summarizes bounded
diagnostics for agent next-step guidance without changing benchmark execution.
**Depends on:** Phase 181
**Requirements:** SIDE-01, SIDE-02, SIDE-03, SIDE-04

**Success Criteria:**
1. `sol_execbench.agent_feedback.v1` models validate bounded status,
   reason-code, bottleneck, recommendation, limitation, authority, and citation
   fields.
2. Evaluation with a trace output path writes
   `trace.jsonl.agent-feedback.json` beside the canonical trace.
3. Sidecar summaries are derived from existing trace/profile/static/stability
   inputs and exclude raw trace rows, raw profiler dumps, full source, prompt
   text, and unstable absolute temp paths.
4. Missing or unavailable optional diagnostic inputs produce explicit
   availability/limitation states and do not fail evaluation.

**Plans:** 2 plans

- [ ] 182-01-PLAN.md -- Agent-feedback sidecar schema, builder, authority
  fields, and unit tests.
- [ ] 182-02-PLAN.md -- CLI/evaluation wiring to persist optional sidecars from
  bounded diagnostic inputs.

### Phase 183: Freshness Identity and Artifact References

**Goal:** HIP can determine whether feedback belongs to the current trace/run
and can cite source artifacts without ingesting raw dumps.
**Depends on:** Phase 182
**Requirements:** IDEN-01, IDEN-02, IDEN-03

**Success Criteria:**
1. Sidecars include trace path, generated timestamp, SOL contract version,
   target/run identity when available, candidate/source hash when available,
   and referenced artifact checksums.
2. Profile, static evidence, environment, stability, and trace-adjacent inputs
   are represented with compact citations.
3. Validator helpers classify stale or mismatched sidecars as diagnostic stale
   states while leaving canonical trace validity unchanged.

**Plans:** 1 plan

- [ ] 183-01-PLAN.md -- Freshness identity model, compact artifact citations,
  checksum wiring, and stale-state validation.

### Phase 184: Governance Guardrails and Compatibility Fixtures

**Goal:** Every feedback state remains diagnostic-only and cannot promote
benchmark authority or release decisions.
**Depends on:** Phase 183
**Requirements:** GOVR-01, GOVR-02, GOVR-03

**Success Criteria:**
1. Schema enforces `diagnostic_only=true` and false authority flags for all
   benchmark and release/cutover authority dimensions.
2. Tests reject contradictory-authority payloads and prove sidecar state cannot
   affect score authority, evidence tier, claim-upgrade status, release gates,
   or cutover eligibility.
3. Public claim-boundary and evidence-quality docs describe the sidecar as
   next-experiment guidance only.

**Plans:** 1 plan

- [ ] 184-01-PLAN.md -- Governance validators, authority guardrails, and
  evidence-quality documentation updates.

### Phase 185: HIP Consumer Integration Package and Docs

**Goal:** HIP Playground can implement v1.26 adapter/runtime work against stable
SOL fixtures and examples.
**Depends on:** Phase 184
**Requirements:** FIXT-01, FIXT-02, FIXT-03

**Success Criteria:**
1. Fixtures cover valid, missing/unavailable, malformed, stale, partial, and
   contradictory-authority sidecar cases.
2. Fixture docs explain mapping SOL bottleneck/recommendation/limitation/citation
   fields into closed consumer taxonomies and safe unknown handling.
3. CPU-safe tests keep example sidecars deterministic and free of raw profiler
   dumps, full source, raw trace rows, and absolute temp paths.

**Plans:** 1 plan

- [ ] 185-01-PLAN.md -- HIP-facing fixtures, example sidecars, mapping notes,
  and deterministic fixture tests.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 181. Feedback Contract and Capability Surface | 0/1 | Not started | - |
| 182. Diagnostic Sidecar Schema and Generator | 0/2 | Not started | - |
| 183. Freshness Identity and Artifact References | 0/1 | Not started | - |
| 184. Governance Guardrails and Compatibility Fixtures | 0/1 | Not started | - |
| 185. HIP Consumer Integration Package and Docs | 0/1 | Not started | - |
