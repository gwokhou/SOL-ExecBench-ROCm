# Documentation

Start with [Getting started](user/GETTING-STARTED.md), then use the
[Cookbook](user/COOKBOOK.md) for common commands or the
[Researcher guide](user/RESEARCHER-GUIDE.md) for experiment-recording
requirements.

## Public contracts and workflows

- [Definition](user/definition.md), [Workload](user/workload.md),
  [Solution](user/solution.md), and [Trace](user/trace.md) define the accepted
  benchmark inputs and canonical result.
- [Schema boundaries](user/schema-boundaries.md) and the
  [Evaluator contract](user/EVALUATOR-CONTRACT.md) describe strict parsing and
  the machine-readable capability surface.
- [Architecture](user/ARCHITECTURE.md), [Configuration](user/CONFIGURATION.md),
  [Cross-path comparison](user/CROSS-PATH-COMPARISON.md),
  and the [SOLAR boundary](SOLAR-BOUNDARY.md) describe process ownership and
  formal analysis.
- [Scoring contract](SCORING-V3.md) and
  [Release scoring](user/RELEASE-SCORING.md) distinguish diagnostic formula
  results from the publisher-authored, content-addressed official-score workflow.
- [Current claim boundaries](user/CLAIMS.md) state exactly what checked-in and
  generated evidence can establish.

## ROCm operation and validation

- [ROCm environment](user/rocm.md)
- [ROCm library categories](user/rocm_libraries.md)
- [Timing and profiling](user/rocm_timing.md)
- [Toolchain routing](user/rocm_toolchain_routing.md)
- [RDNA4 validation scope](user/RDNA4-VALIDATION.md)
- [Testing](user/TESTING.md)

## Diagnostic sidecars

- [Static-kernel evidence](user/static_kernel_evidence.md)
- [Profile summary](user/profile_summary_sidecar.md)
- [Decision sidecar](user/decision_sidecar.md) and its
  [derivation contract](user/decision_sidecar_contract.md)
- [Agent feedback](user/agent_feedback_sidecar.md)

These artifacts are diagnostic. Their guides state the current schema
identifiers and authority limits; none can replace canonical Trace JSONL or
grant official-score authority.

## Policy and maintenance

- [Active follow-ups](../HANDSOFF.md) is the single repository backlog.
- [Provenance](user/provenance.md) and [Compliance](user/compliance.md)
- [Development](user/DEVELOPMENT.md)
- [Architecture navigation](internal/architecture_navigation.md) and
  [coupling governance](internal/coupling_governance.md)
- [AKA task-source research](internal/aka-sol-task-source-case-study.md),
  [corpus friendliness policy](internal/aka-expansion-friendliness.md), and
  [decision-modeling research](internal/decision-modeling-research.md)

Internal research documents record the design basis for policies still
referenced by the implementation. They are not current readiness reports or
release-status claims.

## SOLAR evolution

- [Current roadmap](internal/solar-evolution-roadmap-2026-11-05.md)
- [Next iteration: version baseline, judge discrimination, and LLM_CORE coverage](internal/solar-next-iteration-plan-2026-11-07.md)
- M0 freeze: [baseline evidence](internal/solar-m0-baseline-evidence-2026-11-07.md),
  [judge preregistration](internal/solar-m0-judge-preregistration-2026-11-07.md),
  [LLM_CORE coverage specification](internal/solar-m0-llm-core-coverage-spec-2026-11-07.md)
- [E1/C1 implementation and CPU acceptance, 2026-11-08](internal/solar-e1-c1-cpu-acceptance-2026-11-08.md)
- [B1 execution/environment freeze and CPU baseline](internal/solar-b1-execution-freeze-2026-11-08.md)
- [B1 revised freeze after the CPU audit repair](internal/solar-b1-refreeze-2026-11-08.md)
- [E2 judge discrimination and cost execution ledger](internal/solar-e2-execution-2026-11-08.md)
- [E2 successor development preflight](internal/solar-e2-successor-preflight-2026-11-18.md)
- [E2 successor full-denominator budget and development measurement slots](internal/solar-e2-successor-budget-2026-11-18.md)
- [E2 successor judge discrimination and cost results, 2026-11-19](internal/solar-e2-successor-results-2026-11-19.md)
- [LayerNorm centered-FP64 CPU comparison, dev GPU, and independent holdout results, 2026-11-19](internal/solar-e2-layernorm-holdout-results-2026-11-19.md)
- [B2 acceptance: representative regression and refreshed AKA qualification](internal/solar-b2-acceptance-2026-11-08.md)
- [Historical B2 checkpoint before the approved erratum](internal/solar-b2-checkpoint-2026-11-08.md)
- [Approved B2 representative expectation erratum](internal/solar-b2-representative-erratum-2026-11-08.md)
- [Approved E0 cost-count arithmetic erratum](internal/solar-e0-cost-count-erratum-2026-11-08.md)
- [Stage 0: capabilities, assumptions, and baseline](internal/solar-evolution-stage0.md)
- [Acceptance sample specifications](internal/solar-evolution-acceptance.md)
- [Stage 1–3 task specifications](internal/solar-evolution-tasks.md)

Historical evidence needed to interpret those specifications:

- [BASE-01 original CPU baseline, 2026-10-23](internal/solar-base01-cpu-2026-10-23.md)
- [BASE-01 pre-integration CPU snapshot, 2026-10-26](internal/solar-base01-cpu-2026-10-26.md)
- [Stages 1–5 isolated CPU evidence, 2026-10-26](internal/solar-resume-2026-10-26.md)
- [Historical mapper validation and superseded GPU proposal](internal/solar-mapper-followup-2026-10-26.md)
- [Historical-file retirement decisions, 2026-11-08](internal/history-retirement-2026-11-08.md)

Stages 1–5 are integrated; BASE-01 and the dated stage reports remain historical
snapshots. Current implementation and evidence status are recorded in
[HANDSOFF](../HANDSOFF.md#solar-evolution). The next-iteration document is a plan,
not an execution record or authorization to start implementation or GPU work.
The E1/C1 acceptance record covers the separately authorized implementation and
CPU checks. B1 freezes their execution commit and environment separately;
B2 has passed admission, all 8 representative expectations and the refreshed
AKA static/canary/full chain, with an independent CPU receipt replay. M1 is complete. The original E2 execution stopped at its 20 GiB evidence quota
(325/1,296 correctness sessions), and an initial successor attempt stopped at
PID-limit exhaustion (500/1,296). The PID-reaping successor completed all 432
logical units, 1,296 correctness sessions, 144 cost sessions, and 72 offline
replays; the judge did not meet the overall E0 criteria because 15 LayerNorm
L1 units failed repeat consistency. The opt-in `centered_fp64_v1` LayerNorm
research reference subsequently passed CPU replay (3,669 calls), 156
development GPU sessions, and 186 independent `h0` holdout GPU sessions.
Official scoring contracts remain unchanged, and C2/C3 remain unstarted.

Public contracts in this tree describe the current implementation. Retain dated
reports when they bind unique baseline evidence or acceptance decisions still
referenced by the project. Superseded proposals and unused campaign entry points
are recovered from Git history; their age or lack of incoming links alone does
not establish that they can be removed.
