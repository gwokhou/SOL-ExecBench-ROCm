# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-06-01
**Milestone:** v1.21 Codebase Debt Reduction and Execution Boundary Hardening
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1 Requirements

### Dataset Runner Decomposition

- [x] **DATASET-01**: Maintainer can reason about dataset problem/workload selection through tested package helpers instead of mutable selection logic embedded only in `scripts/run_dataset.py`.
- [x] **DATASET-02**: Maintainer can load and evaluate run-state, resume, rerun, capped workload, ready-subset, and existing-output decisions through deterministic helpers with focused tests.
- [x] **DATASET-03**: Maintainer can construct closure/provenance records through package code that preserves bounded refs, stale-output checks, skipped statuses, and failure classification semantics.
- [x] **DATASET-04**: Maintainer can discover and attach derived evidence refs from trace/output roots without coupling report generation to the main execution loop.
- [x] **DATASET-05**: Existing `scripts/run_dataset.py` CLI behavior, output filenames, trace semantics, and public sidecar contracts remain backward-compatible after decomposition.

### Eval Driver Runtime Decomposition

- [x] **EVAL-01**: Maintainer can test eval-driver helper behavior in package unit tests before staging the generated template.
- [x] **EVAL-02**: Eval driver correctness setup, workload execution helpers, reward-hack check plumbing, and trace/evaluation construction are thinned where practical into importable runtime modules.
- [x] **EVAL-03**: Generated `eval_driver.py` remains responsible only for staging-directory orchestration, dynamic imports, subprocess execution context, and trace emission glue.
- [x] **EVAL-04**: Driver smoke tests continue to prove staged template behavior for PyTorch ROCm, HIP/C++ where available, reward-hack outcomes, and error-status priority.

### Analysis Module Decomposition

- [x] **ANALYSIS-01**: AMD bound graph extraction separates graph construction, operator-family classification, and evidence annotation into smaller helpers with operation-family tests.
- [x] **ANALYSIS-02**: AMD bound estimate formulas are grouped by family or responsibility so changes to one family can be tested without broad scoring regressions.
- [x] **ANALYSIS-03**: SOLAR derivation separates semantic provenance, bound/formula derivation, coverage/status classification, and report rendering while preserving public sidecar schemas.
- [x] **ANALYSIS-04**: Static kernel evidence separates artifact discovery, tool routing, raw-output parsing, bounded capture, and sidecar/report rendering behind focused helpers and fixtures.
- [x] **ANALYSIS-05**: Refactors preserve existing AMD score, AMD SOL/SOLAR, static evidence, and claim-boundary behavior unless a requirement explicitly changes it.

### Execution Boundary Hardening

- [ ] **BOUNDARY-01**: Reward-hack tests cover additional known bypass families with malicious and allowed examples while preserving the documented regex/static-review limits.
- [ ] **BOUNDARY-02**: Clock and timing tests include representative ROCm SMI/device fixture outputs, unsupported/low-power states, and memory-pressure or timing-diagnostic guardrails that do not require real hardware.
- [ ] **BOUNDARY-03**: Static evidence tests cover partial, unavailable, failed, and toolchain-variant parser states using bounded fixtures.
- [ ] **BOUNDARY-04**: Dataset resume and closure tests cover stale traces, stale closure provenance, capped workloads, ready subsets, reruns, missing traces, and derived-evidence combinations.
- [ ] **BOUNDARY-05**: Trace/log and native-build-risk documentation explains what benchmark outputs may contain and when users must use external isolation or scrubbing before publishing.

### Documentation And Claim Guardrails

- [ ] **DOCS-01**: `CONCERNS.md` is updated at milestone completion to distinguish fixed, narrowed, accepted, and externally blocked concerns.
- [ ] **DOCS-02**: Public docs state that v1.21 reduces codebase debt and boundary ambiguity but does not add hard sandboxing, multi-tenant safety, CDNA3/MI300X validation, paper-scale parity, or leaderboard authority.
- [ ] **DOCS-03**: Tests guard against wording that turns diagnostic static evidence, profiler evidence, Docker/container evidence, or local AMD SOL/SOLAR interpretations into stronger claims.
- [ ] **DOCS-04**: Developer documentation explains the new helper/module boundaries for dataset execution, eval driver runtime, scoring derivation, and static evidence.

## Future Requirements

### Hard Sandbox

- **SANDBOX-01**: A future milestone can define a locked-down runner profile for adversarial or multi-tenant submissions with OS/container-level filesystem, network, process, and secret isolation.

### Hardware Validation

- **HW-01**: A future milestone can record real CDNA3/MI300X/CDNA4 or native-host validation evidence gated by v1.20 consistency/stability and v1.21 boundary guardrails.

### Paper And Leaderboard Equivalence

- **PAPER-01**: A future milestone can run paper-scale 235-problem validation and upstream SOLAR comparison only after evidence, hardware, and claim-upgrade prerequisites are satisfied.
- **LEADER-01**: Hosted leaderboard readiness remains future work requiring sandboxing, policy, baseline authority, submission isolation, and operations infrastructure.

### Dependency And Docker Policy

- **DEP-01**: A future milestone can relock per ROCm target or redesign Docker privilege profiles if reproducibility or shared-runner safety becomes more important than current local benchmark ergonomics.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Complete OS/container hard sandbox | Requires runner architecture, policy, and host isolation beyond benchmark helper refactors. |
| CDNA 3, MI300X, CDNA 4, or native-host full-suite validation | Requires physical hardware evidence, not codebase debt reduction alone. |
| Full 235-problem paper-scale validation or upstream SOLAR equivalence | Requires paper-scale runs and comparison evidence outside this debt-focused milestone. |
| Hosted leaderboard or remote submission service | Requires sandboxing, anti-cheat, policy, baseline, and service infrastructure. |
| One-for-one native ROCm replacement proof for all former NVIDIA categories | Requires broader native library workload coverage and hardware validation. |
| Large PyTorch/ROCm dependency relocking or Docker privilege redesign | Deferred unless needed by focused tests or docs in this milestone. |
| Changing canonical Trace, Definition, Workload, Solution, timing, correctness, score, or evaluator contract schemas | v1.21 is a refactor, test, and boundary milestone; public benchmark contracts remain stable. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATASET-01 | Phase 94 | Complete |
| DATASET-02 | Phase 94 | Complete |
| DATASET-03 | Phase 94 | Complete |
| DATASET-04 | Phase 94 | Complete |
| DATASET-05 | Phase 94 | Complete |
| EVAL-01 | Phase 95 | Complete |
| EVAL-02 | Phase 95 | Complete |
| EVAL-03 | Phase 95 | Complete |
| EVAL-04 | Phase 95 | Complete |
| ANALYSIS-01 | Phase 96 | Complete |
| ANALYSIS-02 | Phase 96 | Complete |
| ANALYSIS-03 | Phase 97 | Complete |
| ANALYSIS-04 | Phase 97 | Complete |
| ANALYSIS-05 | Phase 97 | Complete |
| BOUNDARY-01 | Phase 98 | Planned |
| BOUNDARY-02 | Phase 98 | Planned |
| BOUNDARY-03 | Phase 98 | Planned |
| BOUNDARY-04 | Phase 98 | Planned |
| BOUNDARY-05 | Phase 99 | Planned |
| DOCS-01 | Phase 99 | Planned |
| DOCS-02 | Phase 99 | Planned |
| DOCS-03 | Phase 99 | Planned |
| DOCS-04 | Phase 99 | Planned |

**Coverage:**
- v1 requirements: 23 total, 14 complete
- Mapped to phases: 23
- Unmapped: 0

---
*Requirements defined: 2026-06-01*
