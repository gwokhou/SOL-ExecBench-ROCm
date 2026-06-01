# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-06-01
**Active Milestone:** v1.23 Evaluation Reliability and Security Hardening
**Queued Milestone:** v1.24 Dataset Batch Run Trustworthiness
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1.23 Requirements

### Evaluation Failure Diagnostics

- [x] **EVAL-DIAG-01**: No-trace evaluation outcomes persist bounded stdout
  and stderr diagnostics, or equivalent structured evidence, for nonzero
  evaluation failures.
- [x] **EVAL-DIAG-02**: CLI failure messages point users to persisted
  diagnostics without requiring `--verbose` or manual staging inspection.
- [x] **EVAL-DIAG-03**: Regression coverage includes non-JSON stdout, library
  noise, nonzero exits, empty trace output, and canonical trace preservation.

### Staged User Import Isolation

- [x] **EVAL-IMPORT-01**: Python solution sources are imported through
  file-based `importlib` loading with unique generated module identities.
- [x] **EVAL-IMPORT-02**: Supported simple-file and package-style entry paths
  keep their existing behavior while avoiding collisions with `sys.modules`.
- [x] **EVAL-IMPORT-03**: Regression tests prove staged user modules cannot
  resolve to unintended existing modules when names collide.

### Native Compile Option Guardrails

- [x] **COMPILE-GUARD-01**: Dangerous compiler and linker options that can
  reference host paths, response files, dynamic loaders, or unsafe link-time
  behavior are rejected with clear validation errors.
- [x] **COMPILE-GUARD-02**: Documented ROCm/HIP extension compile options used
  by existing examples and tests remain accepted.
- [x] **COMPILE-GUARD-03**: Solution schema and native build-template tests
  cover accepted and rejected option classes.

### Eval Driver Responsibility Boundaries

- [x] **EVAL-BOUNDARY-01**: Correctness, timing, trace emission, source review,
  and reward-hack boundary behavior are exposed through importable helpers
  with focused tests.
- [x] **EVAL-BOUNDARY-02**: The generated eval driver template is reduced to
  staged wiring, subprocess-local setup, and integration glue.
- [x] **EVAL-BOUNDARY-03**: Integrity snapshot coverage remains explicit for
  every benchmark-critical helper reachable from staged execution.
- [x] **EVAL-BOUNDARY-04**: Canonical Trace, Definition, Workload, Solution,
  correctness, timing, score, and evaluator contract schemas remain unchanged
  unless separately approved.

## v1.24 Requirements

### Dataset Reuse Policy Service

- [x] **DATASET-REUSE-01**: Dataset reuse decisions are computed by importable
  helpers with explicit inputs for selected workloads, ready subsets,
  provenance, rerun flags, and output paths.
- [x] **DATASET-REUSE-02**: `scripts/run_dataset.py` delegates reuse and
  stale-provenance policy instead of owning the decision matrix inline.
- [x] **DATASET-REUSE-03**: Tests cover reuse, stale provenance, forced rerun,
  partial-ready, missing-output, and missing-sidecar combinations.

### Dataset Closure And Evidence Completeness

- [x] **DATASET-CLOSURE-01**: Closure records classify missing traces, missing
  timing evidence, missing derived sidecars, skipped workloads, and nonzero CLI
  outcomes without ambiguous success states.
- [x] **DATASET-CLOSURE-02**: Summary, score, timing, and closure references
  are assembled through package-level helpers rather than script-local path
  coupling.
- [x] **DATASET-CLOSURE-03**: Closure reports remain deterministic and
  compatible with existing public sidecar contracts.

### Dataset Failure-Mode Regression Matrix

- [x] **DATASET-REGRESS-01**: Regression fixtures cover stale provenance,
  selected ready subsets, missing derived sidecars, rerun flags, CLI
  timeout/nonzero outcomes, and no-trace outputs.
- [x] **DATASET-REGRESS-02**: Tests distinguish CPU-safe policy behavior from
  live ROCm/GPU execution requirements.
- [x] **DATASET-REGRESS-03**: Dataset runner documentation explains when reuse
  is allowed, blocked, or reported as incomplete.

### Deterministic Dataset Sharding Path

- [x] **DATASET-SHARD-01**: A shard plan divides workloads deterministically
  with stable shard identifiers and one trace file per shard.
- [x] **DATASET-SHARD-02**: Merge rules preserve workload ordering,
  provenance, duplicate detection, and incomplete-shard reporting.
- [x] **DATASET-SHARD-03**: The first implementation or design artifact is
  covered by tests and keeps default dataset CLI behavior compatible.

## Future Requirements

### Hardware Validation

- **HW-01**: A future milestone can record full CDNA3/MI300X/CDNA4 or
  native-host validation on real hardware.

### Native ROCm Example Maturity

- **ROCM-EXAMPLE-01**: A future milestone can replace or validate every former
  NVIDIA library-category placeholder with compiled native ROCm examples and
  live toolchain evidence.

### Hardened Sandbox

- **SANDBOX-01**: A future milestone can design a hardened OS/container runner
  for adversarial or multi-tenant submissions.

### Paper And Leaderboard Equivalence

- **PAPER-01**: A future milestone can run full 235-problem paper-scale
  validation and upstream SOLAR comparison.
- **LEADER-01**: A future milestone can design hosted leaderboard submission
  policy, isolation, and operations.

### Dependency And Docker Policy

- **DEP-01**: A future milestone can perform large PyTorch/ROCm relocking or
  Docker privilege redesign.

### Derived Scoring Modularization

- **SCORING-MOD-01**: A future milestone can split large derived-scoring
  modules by operator family while preserving score-authority boundaries.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full CDNA3/MI300X/CDNA4 validation | Requires real hardware evidence, not code cleanup alone. |
| CDNA 3, MI300X, CDNA 4, or native-host ROCm validation expansion | Remains deferred until real hardware evidence exists. |
| Complete hard sandbox | Requires runner architecture and host isolation work beyond these near-term milestones. |
| Paper-scale parity or leaderboard claims | Requires full validation evidence and policy/infrastructure. |
| Canonical schema changes | These milestones should preserve Trace, Definition, Workload, Solution, timing, correctness, score, and evaluator contract schemas. |
| Large dependency relock | Deferred unless needed for focused consistency guardrails. |
| Derived scoring modularization | Important but structurally separate from evaluation and dataset trustworthiness. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| EVAL-DIAG-01 | Phase 106 | Complete |
| EVAL-DIAG-02 | Phase 106 | Complete |
| EVAL-DIAG-03 | Phase 106 | Complete |
| EVAL-IMPORT-01 | Phase 107 | Complete |
| EVAL-IMPORT-02 | Phase 107 | Complete |
| EVAL-IMPORT-03 | Phase 107 | Complete |
| COMPILE-GUARD-01 | Phase 108 | Complete |
| COMPILE-GUARD-02 | Phase 108 | Complete |
| COMPILE-GUARD-03 | Phase 108 | Complete |
| EVAL-BOUNDARY-01 | Phase 109 | Complete |
| EVAL-BOUNDARY-02 | Phase 109 | Complete |
| EVAL-BOUNDARY-03 | Phase 109 | Complete |
| EVAL-BOUNDARY-04 | Phase 109 | Complete |
| DATASET-REUSE-01 | Phase 110 | Complete |
| DATASET-REUSE-02 | Phase 110 | Complete |
| DATASET-REUSE-03 | Phase 110 | Complete |
| DATASET-CLOSURE-01 | Phase 111 | Complete |
| DATASET-CLOSURE-02 | Phase 111 | Complete |
| DATASET-CLOSURE-03 | Phase 111 | Complete |
| DATASET-REGRESS-01 | Phase 112 | Complete |
| DATASET-REGRESS-02 | Phase 112 | Complete |
| DATASET-REGRESS-03 | Phase 112 | Complete |
| DATASET-SHARD-01 | Phase 113 | Complete |
| DATASET-SHARD-02 | Phase 113 | Complete |
| DATASET-SHARD-03 | Phase 113 | Complete |

**Coverage:**
- v1.23 requirements: 13 total
- v1.23 mapped to phases: 13
- v1.24 requirements: 12 total
- v1.24 mapped to phases: 12
- Unmapped: 0

---
*Requirements defined: 2026-06-01*
