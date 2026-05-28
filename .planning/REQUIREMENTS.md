# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-05-28
**Milestone:** v1.18 ROCm Version Matrix via Docker
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## v1 Requirements

### Matrix Contract

- [x] **MATRIX-01**: The project defines a `sol_execbench.rocm_compatibility_matrix.v1` diagnostic report contract with explicit Target, observed evidence, status, reason-code, artifact, and claim-boundary fields.
- [x] **MATRIX-02**: Each Matrix Entry has a stable **Target** identity that describes the requested validation configuration, including ROCm user-space version, Docker image/tag, PyTorch ROCm Target, validation scope, and intended GPU architecture when known.
- [x] **MATRIX-03**: Matrix reports distinguish requested Target values from observed host, container, Python dependency, toolchain, and GPU evidence.
- [x] **MATRIX-04**: Matrix reports support bounded statuses: `host_validated`, `container_validated`, `mixed_version`, `pytorch_wheel_unavailable`, `runtime_unavailable`, and `not_tested`.
- [x] **MATRIX-05**: Matrix reports include claim flags that keep v1.18 compatibility evidence diagnostic-only and never grant score, paper-parity, or leaderboard authority.
- [x] **MATRIX-06**: `host_validated` is emitted only for direct native-host validation evidence, not for Docker user-space Matrix Entries.

### Docker Matrix

- [x] **DOCKER-01**: The repository provides a checked-in ROCm Docker matrix manifest for declared Targets, initially covering configured `7.0.x`, `7.1.x`, and `7.2.x` logical Matrix Entries where usable Docker image tags are known.
- [x] **DOCKER-02**: `docker/Dockerfile` supports parameterized ROCm base image selection while preserving the current ROCm 7.1 default behavior.
- [x] **DOCKER-03**: `scripts/run_docker.sh` supports selecting a declared ROCm Target and rejects unknown Targets unless an explicit unsafe/untested override is supplied.
- [x] **DOCKER-04**: Docker preflight checks classify missing `/dev/kfd`, missing `/dev/dri`, unsupported Docker context, and inaccessible GPU devices as `runtime_unavailable` before benchmark execution.
- [x] **DOCKER-05**: Docker reports record exact requested image repository/tag and, when available, resolved image digest and build arguments.

### uv And PyTorch ROCm

- [x] **DEPS-01**: Each Matrix Entry records its PyTorch ROCm wheel/index policy, including expected wheel local-version tag and uv index or lock strategy.
- [x] **DEPS-02**: The default project dependency path remains ROCm 7.1 unless a per-Target dependency workflow is explicitly selected and recorded.
- [x] **DEPS-03**: Missing or unsupported PyTorch ROCm wheels are classified as `pytorch_wheel_unavailable`, not as benchmark failures.
- [x] **DEPS-04**: CPU, CUDA, wrong-index, or wrong-ROCm PyTorch wheels are detected from installed package metadata and runtime probes.
- [x] **DEPS-05**: A requested Target whose observed PyTorch ROCm wheel, container ROCm user-space, Triton ROCm package, or toolchain version does not match policy is classified as `mixed_version`.
- [x] **DEPS-06**: Illegal `mixed_version` Targets are blocked during preflight before benchmark execution by default.
- [x] **DEPS-07**: An explicit mixed-version debug override may allow probes or smoke execution to continue, but the resulting entry must remain ineligible for `container_validated`, `host_validated`, score authority, paper-parity authority, or leaderboard authority.

### Runtime Evidence

- [x] **EVID-01**: Compatibility evidence records host-scope ROCm/driver/device-node evidence separately from container-scope ROCm user-space and toolchain evidence.
- [x] **EVID-02**: Compatibility evidence records Python runtime metadata, including `torch.__version__`, `torch.version.hip`, `torch.version.cuda`, PyTorch device availability, and `triton-rocm` status when installed.
- [x] **EVID-03**: Compatibility evidence records GPU metadata, including device count, device name, detected `gfx*` architecture, and visible-device environment variables when available.
- [x] **EVID-04**: The project emits per-Target compatibility JSON and an aggregate compatibility matrix JSON with status counts.
- [x] **EVID-05**: Runtime setup failures, dependency resolution failures, and benchmark correctness/performance results are reported as distinct evidence categories.
- [x] **EVID-06**: Compatibility sidecars do not mutate canonical trace JSONL, correctness semantics, timing semantics, scoring schemas, benchmark defaults, or benchmark exit semantics.

### Documentation And Validation

- [x] **DOCS-01**: Documentation explains that Docker Matrix Entries validate container ROCm user-space on the recorded host driver/devices and do not prove native host ROCm validation.
- [x] **DOCS-02**: Documentation explains Target/requested values versus observed host/container/Python/GPU evidence, including why Target is required for matrix interpretation.
- [x] **DOCS-03**: Documentation states that illegal mixed-version Targets are blocked by default and may only continue under explicit debug override without clean validation claims.
- [x] **DOCS-04**: CPU-safe tests cover status classification, reason-code classification, schema serialization, mixed-version blocking, claim flags, and docs wording guardrails.
- [x] **DOCS-05**: Docker/script tests cover Target selection, default behavior preservation, unknown Target rejection, and command construction without requiring live ROCm hardware.
- [x] **DOCS-06**: Live ROCm validation guidance is marker-gated and records the current host ROCm 7.1.x environment as observed evidence without requiring host reinstall for ROCm 7.0.x or 7.2.x.

## Future Requirements

### Native Host Matrix

- **HOST-01**: Native host validation can be run on separate machines or reinstalled hosts for ROCm 7.0.x, 7.1.x, and 7.2.x.
- **HOST-02**: Native host validation can compare direct host results against Docker user-space results for the same Target.

### Extended Hardware Coverage

- **HW-01**: CDNA 3 and CDNA 4 compatibility Matrix Entries can be marked `host_validated` or `container_validated` only when archived real-hardware evidence exists.
- **HW-02**: Matrix reports can aggregate compatibility status by architecture family after multiple hardware Targets have evidence.

### Matrix Tooling

- **TOOL-01**: Matrix reports can be diffed across runs to highlight status, dependency, image, or runtime changes.
- **TOOL-02**: Compatibility JSON schemas can be exported for external CI or downstream consumers.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Automatic host ROCm driver install, uninstall, or version switching | Host driver management is risky, environment-specific, and outside the Docker-focused v1.18 scope. |
| Claiming Docker success as native host ROCm validation | Docker changes user-space but still depends on the host kernel driver and devices. |
| Full paper dataset validation across every ROCm version | v1.18 is compatibility infrastructure, not paper-scale benchmark validation. |
| Score, timing, correctness, leaderboard, or paper-parity policy changes | Compatibility evidence is diagnostic and must not become benchmark authority. |
| CUDA/NVIDIA compatibility matrix | The project is ROCm-only. |
| Arbitrary unbounded ROCm image tags as normal supported Matrix Entries | The matrix must remain declared, testable, and auditable. |
| CDNA 3/CDNA 4 validation claims without real hardware artifacts | Hardware claims require archived device-specific evidence. |
| CDNA 3 / MI300X real-hardware validation | Deferred until archived hardware-specific full-suite evidence exists. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MATRIX-01 | Phase 78 | Complete |
| MATRIX-02 | Phase 78 | Complete |
| MATRIX-03 | Phase 78 | Complete |
| MATRIX-04 | Phase 78 | Complete |
| MATRIX-05 | Phase 78 | Complete |
| MATRIX-06 | Phase 78 | Complete |
| DOCKER-01 | Phase 79 | Complete |
| DOCKER-02 | Phase 79 | Complete |
| DOCKER-03 | Phase 79 | Complete |
| DOCKER-04 | Phase 79 | Complete |
| DOCKER-05 | Phase 79 | Complete |
| DEPS-01 | Phase 80 | Complete |
| DEPS-02 | Phase 80 | Complete |
| DEPS-03 | Phase 80 | Complete |
| DEPS-04 | Phase 80 | Complete |
| DEPS-05 | Phase 80 | Complete |
| DEPS-06 | Phase 80 | Complete |
| DEPS-07 | Phase 80 | Complete |
| EVID-01 | Phase 81 | Pending |
| EVID-02 | Phase 81 | Pending |
| EVID-03 | Phase 81 | Pending |
| EVID-04 | Phase 81 | Pending |
| EVID-05 | Phase 81 | Pending |
| EVID-06 | Phase 81 | Pending |
| DOCS-01 | Phase 82 | Pending |
| DOCS-02 | Phase 82 | Pending |
| DOCS-03 | Phase 82 | Pending |
| DOCS-04 | Phase 82 | Pending |
| DOCS-05 | Phase 82 | Pending |
| DOCS-06 | Phase 82 | Pending |

**Coverage:**
- v1 requirements: 30 total
- Mapped to phases: 30
- Unmapped: 0

---
*Requirements defined: 2026-05-28*
*Last updated: 2026-05-28 after v1.18 roadmap creation*
