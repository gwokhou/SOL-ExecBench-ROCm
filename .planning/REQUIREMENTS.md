# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-05-31
**Milestone:** v1.19 Research Credibility Without New Hardware
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1 Requirements

### Closure Contracts

- [ ] **CLOS-01**: Dataset execution closure has a strict, sidecar-only core
  contract with deterministic record ordering, totals, source refs, and
  provenance fields.
- [ ] **CLOS-02**: Closure status and reason-code vocabulary distinguishes
  attempted pass, attempted failure, not attempted, filtered, skipped existing
  pass, missing trace, missing derived evidence, and setup/runtime blockers.
- [ ] **CLOS-03**: Closure provenance can detect manifest, readiness,
  ready-subset, workload identity, solution mode, and evidence-requirement
  mismatches before treating existing traces as reusable.
- [ ] **CLOS-04**: Closure helpers expose CPU-safe validation and serialization
  APIs that `scripts/run_dataset.py` can call without changing canonical trace
  JSONL, correctness, timing, or score semantics.

### Paper Denominator Accounting

- [ ] **DENOM-01**: The project emits a `paper_denominator_report.v1` JSON
  sidecar that rolls up the public benchmark denominator by problem, workload,
  category, readiness status, closure status, and evidence gap.
- [ ] **DENOM-02**: Denominator accounting separates ready, blocked,
  unsupported, deferred, evidence-missing, attempted-passed,
  attempted-failed, filtered, skipped, and not-attempted states with stable
  reason codes.
- [ ] **DENOM-03**: Denominator reports reference source manifest, inventory,
  readiness, ready-subset, closure, AMD score, AMD SOL, and SOLAR artifacts by
  path/ref and checksum instead of duplicating full sidecar payloads.
- [ ] **DENOM-04**: Denominator reports include explicit claim-boundary fields
  or equivalent wording that keep paper parity, upstream SOLAR parity,
  leaderboard authority, native host validation, and new hardware validation
  false unless future evidence satisfies those claim-upgrade rules.
- [ ] **DENOM-05**: The project emits a deterministic Markdown summary of the
  denominator report for researcher review, including counts, evidence gaps,
  deferred buckets, and next-evidence hints without claiming validation.

### Compatibility Matrix Tooling

- [ ] **MATRIX-01**: The project can export JSON Schema for
  `MatrixEntry` and `RocmCompatibilityMatrixReport` from the strict Pydantic
  models, including schema identity/version metadata and extra-field behavior.
- [ ] **MATRIX-02**: The project can diff two ROCm Compatibility Matrix reports
  by Target identity and validation scope, reporting added, removed,
  unchanged, and changed entries.
- [ ] **MATRIX-03**: Matrix diffs classify semantic changes across status,
  reason code, requested Target values, observed host/container/Python
  dependency/toolchain/GPU evidence, dependency policy, Docker image metadata,
  clock/evidence metadata, artifact refs, and claim boundaries.
- [ ] **MATRIX-04**: Matrix diff output includes machine-readable JSON and a
  human-readable summary with severity-ranked transitions such as validation
  downgrade, mixed-version drift, runtime unavailability, image/dependency
  drift, GPU architecture drift, and claim-boundary escalation.
- [ ] **MATRIX-05**: Matrix schema export and diff tooling remain diagnostic and
  cannot convert Docker container evidence into native-host validation, score
  authority, paper-parity authority, or leaderboard authority.

### Dataset Runner Hardening

- [ ] **RUNNER-01**: `scripts/run_dataset.py` uses closure helpers to enforce
  ready-subset, readiness, manifest, problem, workload, solution, and evidence
  provenance consistency when resuming or reusing existing output.
- [ ] **RUNNER-02**: Existing passing traces are marked `skipped_existing_pass`
  only when provenance matches the selected run configuration; `--rerun`
  forces reattempt and records fresh closure behavior.
- [ ] **RUNNER-03**: Dataset runner closure output classifies build failures,
  runtime failures, timeouts, nonzero CLI exits, correctness failures, missing
  traces, missing derived evidence, and skipped/unattempted workloads with
  stable reason codes and bounded log refs.
- [ ] **RUNNER-04**: Dataset runner hardening preserves existing default
  execution behavior unless a mismatch, missing evidence, or explicit closure
  option requires a diagnostic stop or sidecar status.
- [ ] **RUNNER-05**: Dataset runner closure writes remain deterministic and
  avoid embedding credentials, proprietary kernels, raw dataset payloads,
  unnecessary absolute paths, or unbounded logs.

### AMD SOL/SOLAR Sanity

- [ ] **SANITY-01**: The project emits an `amd_bound_sanity.v1` diagnostic
  report over existing RDNA 4 and Docker evidence, summarizing AMD SOL/SOLAR
  artifact availability, aggregate statuses, coverage summaries, warnings, and
  evidence gaps.
- [ ] **SANITY-02**: Bound sanity reports distinguish scored, degraded,
  unscored, unsupported, provisional, and missing-evidence states without
  changing AMD-native score semantics or score eligibility rules.
- [ ] **SANITY-03**: Bound sanity reports surface provisional RDNA 4 hardware
  model or model-validation risk and explicitly avoid upstream SOLAR
  equivalence, model-validation, paper-parity, leaderboard, CDNA 3, MI300X,
  CDNA 4, or native-host validation claims.
- [ ] **SANITY-04**: Bound sanity checks consume existing trace, closure,
  AMD SOL, SOLAR derivation, AMD score, and compatibility evidence through
  refs/checksums and do not require new hardware probes, Docker privilege
  changes, or dependency relocking.

### Documentation And Guardrails

- [ ] **DOCS-01**: Documentation explains the v1.19 evidence surfaces:
  denominator report, closure hardening, Matrix schema export, Matrix diff, and
  AMD bound sanity, including how to generate and interpret each artifact.
- [ ] **DOCS-02**: Claim-boundary docs explicitly state that v1.19 does not add
  full 235-problem paper validation, upstream SOLAR parity, score authority,
  leaderboard readiness, CDNA 3/MI300X/CDNA4 validation, or native-host ROCm
  matrix validation.
- [ ] **DOCS-03**: CPU-safe tests cover denominator accounting, closure
  serialization/provenance, Matrix schema export, Matrix diff semantics,
  dataset-runner closure classification, AMD bound sanity reports, and docs
  wording guardrails.
- [ ] **DOCS-04**: Public examples or fixture reports show representative
  JSON/Markdown artifact shapes with bounded logs, relative refs, checksums,
  and explicit authority-false or diagnostic-only interpretation.
- [ ] **DOCS-05**: Existing public contracts remain stable: canonical Trace,
  Definition, Workload, Solution, correctness, timing, score, and evaluator
  contract semantics are unchanged by v1.19 reporting features.

## Future Requirements

### Native Host Matrix

- **HOST-01**: Native host validation can be run on separate machines or
  reinstalled hosts for ROCm 7.0.x, 7.1.x, and 7.2.x.
- **HOST-02**: Native host validation can compare direct host results against
  Docker user-space results for the same Target.

### Extended Hardware Coverage

- **HW-01**: CDNA 3 and CDNA 4 compatibility Matrix Entries can be marked
  `host_validated` or `container_validated` only when archived real-hardware
  evidence exists.
- **HW-02**: Matrix reports can aggregate compatibility status by architecture
  family after multiple hardware Targets have evidence.

### External Evidence And CI

- **EXT-01**: Schema bundles can include additional sidecar contracts beyond
  ROCm Compatibility Matrix payloads after a concrete downstream consumer
  requires them.
- **EXT-02**: CI policy helpers can fail on selected Matrix diff severities
  while keeping benchmark execution semantics unchanged.
- **EXT-03**: Cross-report consistency checks can lint denominator, Matrix,
  closure, score, SOL/SOLAR, and docs artifacts for contradictions.

## Out of Scope

| Feature | Reason |
|---------|--------|
| CDNA 3, MI300X, CDNA 4, or native-host ROCm validation expansion | User explicitly chose not to expand hardware validation in this milestone. |
| Full 235-problem real-hardware validation | v1.19 builds audit/reporting infrastructure, not full paper execution evidence. |
| Claiming denominator accounting as paper parity | Denominator accounting explains status and gaps; it does not validate the full benchmark. |
| Upstream SOLAR equivalence comparison | v1.19 may sanity-check local AMD-derived sidecars only; upstream equivalence remains future work. |
| Hosted leaderboard or remote submission service | Leaderboard readiness requires separate hardware, anti-cheat, baseline, submission, and policy decisions. |
| Canonical trace, scoring, timing, correctness, or evaluator contract changes | v1.19 evidence remains sidecar-only and diagnostic unless a later milestone explicitly changes public contracts. |
| New runtime dependencies, databases, dashboards, or generic deep-diff libraries | Existing Python/Pydantic/JSON/Markdown tooling is sufficient and keeps the milestone low-risk. |
| PyTorch/ROCm dependency relocking or Docker privilege expansion | v1.19 uses existing RDNA 4/Docker evidence surfaces and should not mutate dependency or hardware scope. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLOS-01 | Phase 83 | Pending |
| CLOS-02 | Phase 83 | Pending |
| CLOS-03 | Phase 83 | Pending |
| CLOS-04 | Phase 83 | Pending |
| DENOM-01 | Phase 84 | Pending |
| DENOM-02 | Phase 84 | Pending |
| DENOM-03 | Phase 84 | Pending |
| DENOM-04 | Phase 84 | Pending |
| DENOM-05 | Phase 84 | Pending |
| MATRIX-01 | Phase 85 | Pending |
| MATRIX-02 | Phase 85 | Pending |
| MATRIX-03 | Phase 85 | Pending |
| MATRIX-04 | Phase 85 | Pending |
| MATRIX-05 | Phase 85 | Pending |
| RUNNER-01 | Phase 86 | Pending |
| RUNNER-02 | Phase 86 | Pending |
| RUNNER-03 | Phase 86 | Pending |
| RUNNER-04 | Phase 86 | Pending |
| RUNNER-05 | Phase 86 | Pending |
| SANITY-01 | Phase 87 | Pending |
| SANITY-02 | Phase 87 | Pending |
| SANITY-03 | Phase 87 | Pending |
| SANITY-04 | Phase 87 | Pending |
| DOCS-01 | Phase 88 | Pending |
| DOCS-02 | Phase 88 | Pending |
| DOCS-03 | Phase 88 | Pending |
| DOCS-04 | Phase 88 | Pending |
| DOCS-05 | Phase 88 | Pending |

**Coverage:**
- v1 requirements: 28 total
- Mapped to phases: 28
- Unmapped: 0

---
*Requirements defined: 2026-05-31*
*Last updated: 2026-05-31 after v1.19 roadmap creation*
