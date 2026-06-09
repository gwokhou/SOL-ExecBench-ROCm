# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-06-09
**Milestone:** v1.34 RDNA4 Readiness Blocker Closure
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1 Requirements

Requirements for milestone v1.34. Each requirement maps to exactly one roadmap
phase after roadmap creation.

### Custom Input Evaluation

- [ ] **CUST-01**: Evaluator can execute benchmark-defined custom input
  entrypoints for RDNA4 readiness-blocked workloads using workload axes/scalars
  and the selected ROCm device.
- [ ] **CUST-02**: Evaluator validates generated custom inputs against expected
  keys, tensor/scalar types, dtypes, shapes, and device placement before
  reference or candidate execution.
- [ ] **CUST-03**: Custom input generation is deterministic per workload and
  records enough seed/provenance evidence for repeatable RDNA4 closure runs.
- [ ] **CUST-04**: Custom input failures are classified separately as input
  generation errors or OOM blockers instead of remaining generic
  `readiness_blocked` records.

### Quant Readiness

- [ ] **QUANT-01**: Readiness classification distinguishes real CUDA/NVIDIA
  runtime dependencies from lexical false positives in Quant references,
  comments, class names, and variable names.
- [ ] **QUANT-02**: Quant semantic references that can run under PyTorch ROCm
  are reclassified as ready to attempt or hardware-evidence-needed without
  claiming low-precision hardware authority.
- [ ] **QUANT-03**: True CUDA-only Quant paths remain blocked with precise
  evidence, blocker class, and next action for ROCm porting.
- [ ] **QUANT-04**: Quant readiness reports preserve explicit CDNA4 and
  low-precision hardware-evidence boundaries.

### FlashInfer Readiness

- [ ] **FLASH-01**: FlashInfer-Bench readiness classification separates simple
  PyTorch-compatible workloads from true FlashInfer runtime-dependent
  workloads.
- [ ] **FLASH-02**: PyTorch-compatible FlashInfer-Bench workloads can move out
  of category-wide `flashinfer_runtime_assumption` blocking and become ready to
  attempt on ROCm.
- [ ] **FLASH-03**: Runtime-dependent FlashInfer workloads are classified by
  semantic dependency such as paged decode/prefill, ragged prefill, MLA,
  MoE/FP8 block-scale, or unknown runtime dependency.
- [ ] **FLASH-04**: Residual FlashInfer blockers include evidence and next
  actions without implying performance tuning or full FlashInfer kernel parity.

### RDNA4 Coverage Accounting

- [ ] **COV-01**: RDNA4 coverage recompute preserves the 235-problem
  denominator while showing before/after transitions for all 114 original
  readiness-blocked problems.
- [ ] **COV-02**: Coverage and blocker ledgers distinguish resolved readiness
  blockers from new runtime, OOM, correctness, profiler, hardware-evidence, or
  residual readiness blockers.
- [ ] **COV-03**: CPU-safe regression tests prevent readiness blockers from
  being dropped, double-counted, or counted as profiler-backed timing or passed
  validation.

### Claim Guardrails

- [ ] **CLAIM-01**: Reports and docs state that readiness blocker reduction is
  not validation success unless execution evidence passes.
- [ ] **CLAIM-02**: Public and internal claim guardrails prevent v1.34 RDNA4
  readiness evidence from upgrading paper-parity, leaderboard, CDNA3, or CDNA4
  claims.
- [ ] **CLAIM-03**: Documentation records residual blocker classes and deferred
  boundaries for any original readiness-blocked problems not safely executable
  by milestone close.

## v2 Requirements

Deferred to future milestones.

### Performance and Broader Hardware

- **PERF-01**: Project can optimize FlashInfer-equivalent ROCm kernels for
  performance after semantic readiness is established.
- **PERF-02**: Project can collect larger-memory AMD GPU evidence for workloads
  that remain blocked by current RDNA4 16GB memory limits.
- **HW-01**: Project can validate CDNA3 or CDNA4 claims only from complete
  architecture-specific hardware evidence chains.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Random substitution for custom inputs | Breaks benchmark semantics and would produce invalid correctness evidence. |
| Full paper-parity or hosted leaderboard claim | v1.34 only reduces readiness blockers and preserves existing authority boundaries. |
| CDNA3 or CDNA4 validation claim upgrade | RDNA4 readiness evidence cannot validate other architecture families. |
| High-performance FlashInfer kernel tuning | Milestone scope is semantic readiness and blocker closure, not performance parity. |
| Public redistribution of NVIDIA/SOL-ExecBench dataset payloads | Existing local-only dataset and license boundaries remain in force. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
| --- | --- | --- |
| CUST-01 | TBD | Pending |
| CUST-02 | TBD | Pending |
| CUST-03 | TBD | Pending |
| CUST-04 | TBD | Pending |
| QUANT-01 | TBD | Pending |
| QUANT-02 | TBD | Pending |
| QUANT-03 | TBD | Pending |
| QUANT-04 | TBD | Pending |
| FLASH-01 | TBD | Pending |
| FLASH-02 | TBD | Pending |
| FLASH-03 | TBD | Pending |
| FLASH-04 | TBD | Pending |
| COV-01 | TBD | Pending |
| COV-02 | TBD | Pending |
| COV-03 | TBD | Pending |
| CLAIM-01 | TBD | Pending |
| CLAIM-02 | TBD | Pending |
| CLAIM-03 | TBD | Pending |

**Coverage:**
- v1 requirements: 18 total
- Mapped to phases: 0
- Unmapped: 18

---
*Requirements defined: 2026-06-09*
*Last updated: 2026-06-09 after v1.34 requirements definition*

