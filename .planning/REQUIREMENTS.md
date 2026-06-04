# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-06-04
**Milestone:** v1.29 Dataset Migration and Compliance
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1.29 Requirements

### Dataset License And Provenance

- [x] **DATA-LIC-01**: Maintainer can distinguish `nvidia/SOL-ExecBench`,
  `flashinfer-ai/flashinfer-trace`, generated local migration artifacts, and
  project-owned ROCm code through machine-readable provenance metadata.
- [x] **DATA-LIC-02**: Maintainer can identify which dataset artifacts may be
  committed, published, redistributed, generated locally, or excluded from
  release bundles.
- [x] **DATA-LIC-03**: Compliance checks fail if NVIDIA/SOL-ExecBench original
  or derivative dataset content is staged for repository or release
  redistribution.
- [x] **DATA-LIC-04**: Documentation preserves required license and attribution
  boundaries for NVIDIA Evaluation Dataset License content and Apache-2.0
  FlashInfer Trace content.

### Local Dataset Migration Pipeline

- [x] **DATA-MIG-01**: Maintainer can run a local migration command that
  converts downloaded SOL-ExecBench rows into the repository's benchmark
  problem layout without committing source dataset content.
- [x] **DATA-MIG-02**: Maintainer can run a local migration command that
  normalizes FlashInfer Trace definitions, workloads, solutions, and traces into
  ROCm runner-compatible local inputs.
- [x] **DATA-MIG-03**: Migration outputs include deterministic manifests,
  checksums, source dataset identifiers, source revisions, and license-boundary
  metadata.
- [x] **DATA-MIG-04**: Migration tooling handles absent optional blobs,
  safetensors inputs, traces, and solution records with explicit blocker states
  instead of silent omission.

### ROCm Readiness Classification

- [x] **DATA-READY-01**: Maintainer can classify migrated workloads as
  PyTorch-compatible, ROCm-port-needed, FlashInfer-specific,
  NVFP4/Blackwell-specific, unsupported, or blocked by missing evidence.
- [x] **DATA-READY-02**: Maintainer can generate a ready subset for workloads
  that are safe to attempt on the current ROCm runner, with denominator and
  exclusion reasons preserved.
- [x] **DATA-READY-03**: Readiness reports expose migration blockers for CUDA
  kernel dependencies, FlashInfer-specific runtime assumptions, NVIDIA-specific
  low-precision formats, missing workload blobs, and unsupported dtypes.
- [x] **DATA-READY-04**: CPU-safe tests prove readiness classification does not
  upgrade blocked, unvalidated, or hardware-specific workloads into validation
  claims.

### NVIDIA/Blackwell Low-Precision ROCm Equivalence

- [x] **LOWP-ROCM-01**: Maintainer can import ROCm compatibility abstractions
  for NVFP4/Blackwell-specific dataset semantics without requiring CDNA4
  hardware.
- [x] **LOWP-ROCM-02**: Compatibility implementations preserve public entry
  points, tensor shapes, packing/unpacking semantics, scale metadata, and
  reference behavior needed by migrated benchmark definitions.
- [x] **LOWP-ROCM-03**: Compatibility paths emit explicit unvalidated-CDNA4
  evidence markers so they cannot be mistaken for performance or hardware
  validation.
- [x] **LOWP-ROCM-04**: Tests cover semantic round trips and blocker behavior
  on CPU-safe or mocked ROCm paths while deferring real CDNA4 execution.

### Dataset Runner Integration

- [x] **DATA-RUN-01**: `scripts/run_dataset.py` can consume locally migrated
  dataset roots, ready subsets, manifests, and license-boundary metadata without
  changing existing single-problem behavior.
- [x] **DATA-RUN-02**: Execution closure records include migration source,
  checksum, license-boundary, readiness, and blocker evidence for each selected
  workload.
- [x] **DATA-RUN-03**: Reuse and rerun decisions treat migration manifest,
  readiness classification, solution mode, and requested evidence changes as
  provenance drift.
- [x] **DATA-RUN-04**: Dataset runner reports skipped, missing, blocked, and
  unvalidated low-precision workloads in the denominator rather than silently
  dropping them.

### Documentation And Guardrails

- [x] **DATA-DOC-01**: Documentation explains how to download source datasets,
  run local migration, generate ready subsets, and execute bounded ROCm runs.
- [x] **DATA-DOC-02**: Public docs state that NVIDIA/SOL-ExecBench content is
  not redistributed by this project and must be obtained and migrated locally by
  users with appropriate license rights.
- [x] **DATA-DOC-03**: Docs distinguish ROCm-compatible dataset migration,
  low-precision semantic implementation, real CDNA3/CDNA4 validation, paper
  parity, and leaderboard authority.
- [x] **DATA-DOC-04**: Release and prerelease guardrails fail if migrated
  NVIDIA dataset content is included in release bundles or if unvalidated
  low-precision compatibility is described as hardware-validated.

## Future Requirements

Deferred to later milestones because they require hardware access, broader
kernel optimization scope, or operational infrastructure outside v1.29.

### Hardware Validation

- **HWVAL-CDNA3-01**: Validator can execute the migrated ready subset on real
  CDNA3 hardware with complete environment, timing, closure, and score evidence.
- **HWVAL-CDNA4-01**: Validator can execute NVFP4/Blackwell-equivalent
  compatibility paths on real CDNA4 hardware with correctness and performance
  evidence.

### Kernel Optimization

- **FLASH-KERNEL-01**: Developer can port and tune FlashInfer CUDA kernels to
  HIP, Triton ROCm, CK, or another ROCm backend with performance comparison
  against appropriate baselines.
- **LOWP-PERF-01**: Developer can optimize low-precision ROCm compatibility
  paths and publish bounded performance evidence after real hardware validation.

### Research Authority

- **PAPER-235-01**: Validator can run and report complete 235-problem paper-scale
  validation on supported AMD hardware with paper-denominator accounting.
- **LEADERBOARD-01**: Operator can host or publish benchmark results through a
  leaderboard or remote submission service.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Publishing or hosting NVIDIA/SOL-ExecBench original or derivative dataset content | NVIDIA dataset license is not open-source and restricts redistribution. |
| Claiming complete 235-problem ROCm or paper parity | Requires full paper-scale validation outside this migration milestone. |
| Real CDNA4 validation or performance authority for NVFP4/Blackwell semantics | Hardware evidence is not available in this milestone. |
| High-performance FlashInfer CUDA-kernel ROCm tuning and full performance comparison | Separate backend optimization effort; v1.29 focuses on migration, compatibility, and blockers. |
| Real CDNA3 or CDNA4 full-suite execution | Requires complete hardware evidence chain outside this local planning scope. |
| Hosted leaderboard or remote submissions | Operational scope remains deferred. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-LIC-01 | Phase 131 | Complete |
| DATA-LIC-02 | Phase 131 | Complete |
| DATA-LIC-03 | Phase 131 | Complete |
| DATA-LIC-04 | Phase 131 | Complete |
| DATA-MIG-01 | Phase 132 | Complete |
| DATA-MIG-02 | Phase 132 | Complete |
| DATA-MIG-03 | Phase 132 | Complete |
| DATA-MIG-04 | Phase 132 | Complete |
| DATA-READY-01 | Phase 133 | Complete |
| DATA-READY-02 | Phase 133 | Complete |
| DATA-READY-03 | Phase 133 | Complete |
| DATA-READY-04 | Phase 133 | Complete |
| LOWP-ROCM-01 | Phase 134 | Complete |
| LOWP-ROCM-02 | Phase 134 | Complete |
| LOWP-ROCM-03 | Phase 134 | Complete |
| LOWP-ROCM-04 | Phase 134 | Complete |
| DATA-RUN-01 | Phase 135 | Complete |
| DATA-RUN-02 | Phase 135 | Complete |
| DATA-RUN-03 | Phase 135 | Complete |
| DATA-RUN-04 | Phase 135 | Complete |
| DATA-DOC-01 | Phase 135 | Complete |
| DATA-DOC-02 | Phase 135 | Complete |
| DATA-DOC-03 | Phase 135 | Complete |
| DATA-DOC-04 | Phase 135 | Complete |

**Coverage:**
- v1.29 requirements: 24 total
- Mapped to phases: 24
- Unmapped: 0

---
*Requirements defined: 2026-06-04*
*Last updated: 2026-06-04 after Phase 135 completion*
