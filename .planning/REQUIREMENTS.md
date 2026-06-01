# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-06-01
**Milestone:** v1.22 Concern Closure and Execution Boundary Hardening
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1 Requirements

### Dataset Runner Closure

- [ ] **DATASET-01**: Maintainer can invoke dataset problem execution through an importable runner abstraction instead of direct subprocess orchestration embedded only in `scripts/run_dataset.py`.
- [ ] **DATASET-02**: Maintainer can construct solution wrapping and reference/custom Python source handling without global text replacement that mutates strings, comments, or legitimate identifiers.
- [ ] **DATASET-03**: Maintainer can write dataset summaries, score reports, timing evidence refs, and closure reports through package helpers with focused tests.
- [ ] **DATASET-04**: Dataset-scale runs preserve existing CLI behavior while exposing a safe seam for future scheduling or bounded CPU-side parallel report work.

### Eval Driver Diagnostics

- [ ] **EVAL-01**: Maintainer can test reference timing behavior through importable helpers without staging the full generated driver.
- [ ] **EVAL-02**: When reference benchmarking is requested and reference timing fails, traces, logs, or status semantics expose that failure explicitly instead of silently leaving `reference_latency_ms` at `0.0`.
- [ ] **EVAL-03**: Trace emission and stdout/stderr framing are covered by regression tests that prove user prints and noisy imports cannot corrupt JSONL output.
- [ ] **EVAL-04**: Correctness/timing orchestration remains benchmark-compatible while moving avoidable pure logic out of `eval_driver.py`.

### Source Review And Boundary Evidence

- [ ] **BOUNDARY-01**: Source-review tests cover additional process, file, import, native loader, stream, cache, and obfuscation bypass families.
- [ ] **BOUNDARY-02**: Python source review has an AST-aware or token-aware path for cases where regex scanning is too broad or too easy to bypass.
- [ ] **BOUNDARY-03**: Blocked or flagged source-review outcomes are represented as structured evidence in traces, sidecars, or logs without implying hard sandboxing.
- [ ] **BOUNDARY-04**: Public and developer docs clearly state that static review plus subprocess execution is not a hardened multi-tenant sandbox.

### Scoring And Evidence Fixtures

- [ ] **SCORING-01**: SOLAR and AMD bound derivation have family-specific golden fixtures for representative operator families and fallback behavior.
- [ ] **SCORING-02**: Confidence/status transitions in SOLAR and AMD bound derivation are covered by focused tests independent of broad report shape tests.
- [ ] **SCORING-03**: Static kernel evidence can consume or produce an explicit artifact manifest when build outputs are known, reducing reliance on recursive build-tree scanning.
- [ ] **SCORING-04**: Static evidence and derived scoring changes preserve diagnostic-only authority and existing public sidecar contracts.

### Dependency And Closure Guardrails

- [ ] **GUARD-01**: ROCm wheel, Docker target, and dependency-matrix policy consistency is guarded by tests or docs checks when target metadata changes.
- [ ] **GUARD-02**: Dataset closure provenance tests cover new sidecar refs, stale provenance combinations, and manifest/cache provenance behavior.
- [ ] **GUARD-03**: Hardware-marker skip behavior remains explicit so CPU-safe green runs cannot be mistaken for RDNA4/CDNA3/timing validation.

### Concern Map Stewardship

- [ ] **DOCS-01**: `CONCERNS.md` preserves milestone-management context for recently narrowed, still actionable, accepted, and externally deferred concerns.
- [ ] **DOCS-02**: v1.22 completion updates `CONCERNS.md` so each in-scope item is marked fixed, narrowed, or carried forward with evidence.
- [ ] **DOCS-03**: Out-of-scope items remain explicit: CDNA3/MI300X/CDNA4 validation, paper-scale parity, leaderboard readiness, and complete hard sandboxing.

## Future Requirements

### Hardware Validation

- **HW-01**: A future milestone can record full CDNA3/MI300X/CDNA4 or native-host validation on real hardware.

### Hardened Sandbox

- **SANDBOX-01**: A future milestone can design a hardened OS/container runner for adversarial or multi-tenant submissions.

### Paper And Leaderboard Equivalence

- **PAPER-01**: A future milestone can run full 235-problem paper-scale validation and upstream SOLAR comparison.
- **LEADER-01**: A future milestone can design hosted leaderboard submission policy, isolation, and operations.

### Dependency And Docker Policy

- **DEP-01**: A future milestone can perform large PyTorch/ROCm relocking or Docker privilege redesign.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full CDNA3/MI300X/CDNA4 validation | Requires real hardware evidence, not code cleanup alone. |
| Complete hard sandbox | Requires runner architecture and host isolation work beyond this milestone. |
| Paper-scale parity or leaderboard claims | Requires full validation evidence and policy/infrastructure. |
| Canonical schema changes | This milestone should preserve Trace, Definition, Workload, Solution, timing, correctness, score, and evaluator contract schemas. |
| Large dependency relock | Deferred unless needed for focused consistency guardrails. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATASET-01 | Phase 100 | Pending |
| DATASET-02 | Phase 100 | Pending |
| DATASET-03 | Phase 100 | Pending |
| DATASET-04 | Phase 100 | Pending |
| EVAL-01 | Phase 101 | Pending |
| EVAL-02 | Phase 101 | Pending |
| EVAL-03 | Phase 101 | Pending |
| EVAL-04 | Phase 101 | Pending |
| BOUNDARY-01 | Phase 102 | Pending |
| BOUNDARY-02 | Phase 102 | Pending |
| BOUNDARY-03 | Phase 102 | Pending |
| BOUNDARY-04 | Phase 102 | Pending |
| SCORING-01 | Phase 103 | Pending |
| SCORING-02 | Phase 103 | Pending |
| SCORING-03 | Phase 103 | Pending |
| SCORING-04 | Phase 103 | Pending |
| GUARD-01 | Phase 104 | Pending |
| GUARD-02 | Phase 104 | Pending |
| GUARD-03 | Phase 104 | Pending |
| DOCS-01 | Phase 105 | Pending |
| DOCS-02 | Phase 105 | Pending |
| DOCS-03 | Phase 105 | Pending |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0

---
*Requirements defined: 2026-06-01*
