---
gsd_state_version: 1.0
milestone: v1.22
milestone_name: Concern Closure and Execution Boundary Hardening
status: planning
last_updated: "2026-06-01T04:45:19.701Z"
last_activity: 2026-06-01
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-01)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.
**Current focus:** v1.21 codebase debt reduction and execution boundary hardening

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-06-01 — Milestone v1.22 started

## Performance Metrics

**Velocity:**

- Total plans completed: 10
- Average duration: n/a
- Total execution time: 0h

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 89. Cross-Report Consistency Contract And Lint | 2/2 | Complete | n/a |
| 90. Evaluation Stability Evidence | 2/2 | Complete | n/a |
| 91. Claim Upgrade Rules And Authority Gates | 2/2 | Complete | n/a |
| 92. Trust Summary Integration | 2/2 | Complete | n/a |
| 93. Documentation, Examples, And Guardrail Tests | 2/2 | Complete | n/a |

**Recent Trend:**

- Last milestone: v1.19 shipped Phases 83-88 on 2026-05-31.
- Trend: v1.20 starts with five standard-granularity phases focused on cross-report consistency, timing-quality diagnostics, claim-upgrade rules, trust summaries, and guardrails before new hardware validation.

| Phase 83 P01 | 403s | 2 tasks | 2 files |
| Phase 83 P02 | 403s | 3 tasks | 4 files |
| Phase 84 P01 | 352 | 3 tasks | 2 files |
| Phase 84 P02 | 263 | 3 tasks | 4 files |
| Phase 85 P01 | 215 | 3 tasks | 3 files |
| Phase 85 P02 | 491 | 3 tasks | 4 files |
| Phase 86 P01 | 358 | 3 tasks | 4 files |
| Phase 86 P02 | 185 | 3 tasks | 4 files |
| Phase 87 P02 | 265 | 3 tasks | 3 files |
| Phase 87 P01 | 1069 | 3 tasks | 2 files |
| Phase 88 P01 | 960 | 3 tasks | 5 files |
| Phase 88 P02 | 780 | 2 tasks | 11 files |

## Accumulated Context

### Decisions

- v1.19 starts at Phase 83 because v1.18 completed Phases 78-82.
- v1.19 derives phases only from v1.19 requirements: closure, denominator, Matrix tooling, runner hardening, AMD bound sanity, and docs/tests.
- v1.19 must not expand CDNA3, MI300X, CDNA4, or native-host hardware validation.
- v1.19 evidence remains sidecar/reporting infrastructure and must not change canonical Trace, Definition, Workload, Solution, correctness, timing, scoring, or evaluator semantics.
- Denominator, Matrix, Docker, and AMD SOL/SOLAR reports must keep paper parity, score authority, leaderboard authority, native-host validation, and new-hardware validation claims false unless future evidence upgrades them.
- [Phase 83]: scripts/run_dataset.py delegates execution closure status, totals, record validation, report construction, and writing to core helpers without adding resume/reuse enforcement.
- [Phase 83]: Execution closure remains a sidecar-only sol_execbench.execution_closure.v1 contract.
- [Phase 84]: [Phase 84 Plan 01]: Paper denominator accounting remains a sidecar-only report and does not change canonical benchmark schemas.
- [Phase 84]: [Phase 84 Plan 01]: Missing evidence is accounted as evidence_missing/deferred and never upgraded into validation, score, leaderboard, paper, SOLAR, native-host, or new-hardware authority.
- [Phase 84]: [Phase 84 Plan 02]: Optional AMD SOL and SOLAR artifacts remain bounded source refs/checksums, not embedded payloads.
- [Phase 84]: [Phase 84 Plan 02]: Paper denominator reports are exposed through a script and dataset helper exports, not primary sol-execbench CLI options.
- [Phase 85]: [Phase 85 Plan 01]: Schema export remains script-side diagnostic tooling, not a primary sol-execbench CLI option.
- [Phase 85]: [Phase 85 Plan 01]: Matrix schema exports are limited to MatrixEntry and RocmCompatibilityMatrixReport.
- [Phase 85]: [Phase 85 Plan 02]: Diff JSON and Markdown remain diagnostic-only with score, paper-parity, leaderboard, and native-host validation authority false.
- [Phase 85]: [Phase 85 Plan 02]: Matrix report diffs match entries by target_id plus validation_scope.
- [Phase 86]: Raw CLI stdout/stderr remains in per-problem log files while execution_closure.json stores relative cli_log_ref plus concise notes. — Keeps closure sidecars bounded and avoids embedding raw logs or absolute temp paths.
- [Phase 86]: Existing passing traces authorize skipped_existing_pass only when output/execution_closure.json contains matching provenance. — Prevents stale or tampered sidecars from authorizing clean reuse.
- [Phase 87]: [Phase 87 Plan 02]: AMD bound sanity generation is exposed only as a research script, not as a primary sol-execbench CLI option or package entry point.
- [Phase 87]: [Phase 87 Plan 02]: The report script loads only explicitly supplied JSON paths and delegates normalization, checksum, status, and rendering behavior to core helpers.
- [Phase 87]: [Phase 87 Plan 01]: amd_bound_sanity.v1 remains a scoring sidecar/reporting artifact and does not modify score eligibility or canonical schemas.
- [Phase 87]: [Phase 87 Plan 01]: Claim boundaries are literal false fields plus visible Markdown wording; provisional RDNA 4 risk is a risk flag, not validation.
- [Phase 88]: [Phase 88]: v1.19 evidence guidance is centralized in docs/v1_19_evidence_guide.md and linked from CLAIMS, TESTING, and RESEARCHER-GUIDE.
- [Phase 88]: [Phase 88]: v1.19 fixtures remain demo-only with synthetic checksums, relative refs, bounded log refs, diagnostic-only wording, and false authority fields.
- v1.20 starts at Phase 89 because v1.19 completed Phases 83-88.
- v1.20 derives phases only from v1.20 requirements: consistency lint, evaluation stability, claim-upgrade rules, trust summary integration, and docs/tests.
- v1.20 must not expand full 235-problem paper validation, CDNA3/MI300X/CDNA4 validation, native-host Matrix authority, hosted leaderboard readiness, or upstream SOLAR parity.
- v1.20 evidence remains sidecar/reporting infrastructure and must not change canonical Trace, Definition, Workload, Solution, correctness, timing, scoring, or evaluator semantics.
- Consistency, stability, claim-upgrade, and trust summary artifacts must remain diagnostic until future evidence satisfies explicit claim-upgrade prerequisites.
- [Phase 89]: consistency_report.v1 remains a sidecar/reporting artifact and does not modify public CLI, Trace, Workload, Definition, timing, scoring, or evaluator contracts.
- [Phase 89]: Consistency lint treats cross-report contradictions as diagnostic blockers/warnings; it does not upgrade score, paper-parity, leaderboard, native-host, or new-hardware authority.
- [Phase 90]: evaluation_stability.v1 remains sidecar-only and summarizes timing quality without changing canonical timing, score, correctness, trace, or evaluator semantics.
- [Phase 90]: Stability classification supports interpretation only; it keeps correctness, score, paper-parity, leaderboard, native-host, and new-hardware authority false.
- [Phase 91]: claim_upgrade.v1 evaluates prerequisites and next evidence only; it never mutates authority fields in source reports.
- [Phase 91]: Claim levels are explicit and remain blocked when consistency, stability, denominator, score, Matrix, or hardware-validation evidence is missing or contradictory.
- [Phase 92]: trust_summary.v1 combines evidence-quality sidecars into bounded review guidance with source refs/checksums only.
- [Phase 92]: Trust summaries provide next steps for future CDNA3/MI300X/native-host/paper-scale validation without claiming those validations were performed.
- [Phase 93]: v1.20 evidence-quality docs and fixtures are demo-only, bounded, and explicitly negative on paper, leaderboard, native-host, and new-hardware authority.
- [Phase 93]: Public contract guardrails cover v1.20 consistency, stability, claim-upgrade, and trust-summary sidecars.
- v1.21 starts at Phase 94 because v1.20 completed Phases 89-93.
- v1.21 derives phases only from `.planning/codebase/CONCERNS.md` concerns that can be resolved through code, tests, docs, and repeatable local verification.
- v1.21 must not claim hard sandboxing, multi-tenant safety, CDNA3/MI300X/CDNA4 validation, paper-scale SOLAR parity, hosted leaderboard readiness, or one-for-one native ROCm replacement proof for every former NVIDIA category.
- v1.21 refactors and tests must preserve canonical Trace, Definition, Workload, Solution, timing, correctness, score, and evaluator contract schemas.
- [Phase 94]: Dataset runner selection/run-state and closure/provenance helpers moved into `sol_execbench.core.dataset.run_state` and `sol_execbench.core.dataset.run_closure`; `scripts/run_dataset.py` retains thin compatibility wrappers and CLI orchestration.
- [Phase 95]: Eval-driver staging and import setup moved into `sol_execbench.core.bench.eval_runtime`; `eval_driver.py` remains the subprocess integration shell for timing, trace emission, and execution-context behavior.
- [Phase 96]: AMD bound graph classification and estimate dispatch taxonomy moved into focused helper modules; public bound graph and estimate schemas remain unchanged.
- [Phase 97]: SOLAR derivation status/source-boundary helpers and static evidence extractor aggregation helpers were split out with focused tests; diagnostic-only sidecar authority remains unchanged.
- [Phase 98]: Added CPU-safe boundary tests for reward-hack bypass families, unsupported ROCm SMI clock output, static evidence aggregate states, and dataset derived-evidence closure combinations.
- [Phase 99]: Public claims/developer docs and `CONCERNS.md` now separate v1.21 narrowed debt from externally deferred hard sandbox, hardware-validation, paper-scale parity, and leaderboard work.

### Pending Todos

None found.

### Blockers/Concerns

None.

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Hardware validation | CDNA 3, MI300X, CDNA 4, and native-host ROCm validation expansion | Deferred | v1.19 scope |
| Paper parity | Full 235-problem real-hardware validation and upstream SOLAR parity | Deferred | v1.19 scope |
| Public service | Hosted leaderboard or remote submission service | Deferred | v1.19 scope |
| Dependencies | PyTorch/ROCm dependency relocking or Docker privilege expansion | Deferred | v1.19 scope |
| Hardware validation | CDNA 3, MI300X, CDNA 4, and native-host ROCm validation expansion | Deferred | v1.20 scope |
| Paper parity | Full 235-problem real-hardware validation and upstream SOLAR parity | Deferred | v1.20 scope |
| Public service | Hosted leaderboard or remote submission service | Deferred | v1.20 scope |
| Dependencies | Dependency relocking, Docker privilege expansion, databases, dashboards, or remote services | Deferred | v1.20 scope |
| Security | Complete hard sandbox and multi-tenant/adversarial submission isolation | Deferred | v1.21 scope |
| Hardware validation | CDNA 3, MI300X, CDNA 4, and native-host full-suite validation | Deferred | v1.21 scope |
| Paper parity | Full 235-problem paper-scale validation and upstream SOLAR equivalence | Deferred | v1.21 scope |
| Public service | Hosted leaderboard or remote submission service | Deferred | v1.21 scope |

### Candidate Next-Milestone Frontiers

- Hardware validation expansion: CDNA3/MI300X or CDNA4 live validation using
  v1.20 consistency and stability artifacts as gates.

- Paper parity: full 235-problem validation readiness with claim-upgrade checks
  before any paper-parity candidate claim.

- Profiling diagnostics: feed ROCm profiler or Instruction Roofline evidence
  into stability and trust summaries.

- Leaderboard readiness: submission isolation, anti-cheat, baseline policy, and
  claim-gated trust summaries.

## Quick Tasks Completed

| Date | Quick Task | Result |
|------|------------|--------|
| 2026-05-31 | 260531-u2s add requires_rocm coverage for CLI and dataset runner GPU paths | Added 3 ROCm E2E regressions; `requires_rocm` now reports 17 passed. |
| 2026-05-31 | 260531-uki add remaining requires_rocm e2e coverage | Added HIP/C++ CLI, static evidence, and run_dataset reuse/rerun E2E regressions; `requires_rocm` now reports 19 passed. |
| 2026-05-31 | 260531-rdf add run_dataset closure e2e gaps | Added filtered, missing-workload, readiness-blocked, and stale-provenance ROCm E2E regressions; `requires_rocm` now reports 21 passed. |
| 2026-05-31 | 20260531-fix-v120-audit-gaps | Fixed v1.20 audit blockers; v1.20 aggregate CPU-safe suite reports 74 passed and audit status is passed. |
| 2026-05-31 | 260531-x41 fix v1.20 audit doc wiring tech debt | Added AMD SOL/SOLAR refs to the consistency guide command, required them in docs tests, and moved the audit status to passed. |
| 2026-06-01 | 260601-q06 finish remaining codebase quick debt | Added CLI packager cleanup at known exits, extracted eval output invocation, extended reward-hack dynamic process blocking, and updated concerns. |

## Session Continuity

Last session: 2026-06-01T11:58:02+08:00
Stopped at: Phase 99 complete
Resume file: None

## Operator Next Steps

- Start the next milestone with `$gsd-new-milestone`
