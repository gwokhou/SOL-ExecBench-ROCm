# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-05-28
**Milestone:** v1.18 ROCm Version Matrix via Docker
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## v1 Requirements

### Matrix Contract

- [ ] **MATRIX-01**: The project defines a `sol_execbench.rocm_compatibility_matrix.v1` diagnostic report contract with explicit target, observed evidence, status, reason-code, artifact, and claim-boundary fields.
- [ ] **MATRIX-02**: Each matrix entry has a stable **Target** identity that describes the requested validation configuration, including ROCm user-space version, Docker image/tag, PyTorch ROCm target, validation scope, and intended GPU architecture when known.
- [ ] **MATRIX-03**: Matrix reports distinguish requested target values from observed host, container, Python dependency, toolchain, and GPU evidence.
- [ ] **MATRIX-04**: Matrix reports support bounded statuses: `host_validated`, `container_validated`, `mixed_version`, `pytorch_wheel_unavailable`, `runtime_unavailable`, and `not_tested`.
- [ ] **MATRIX-05**: Matrix reports include claim flags that keep v1.18 compatibility evidence diagnostic-only and never grant score, paper-parity, or leaderboard authority.
- [ ] **MATRIX-06**: `host_validated` is emitted only for direct native-host validation evidence, not for Docker user-space rows.

### Docker Matrix

- [ ] **DOCKER-01**: The repository provides a checked-in ROCm Docker matrix manifest for declared targets, initially covering configured `7.0.x`, `7.1.x`, and `7.2.x` logical rows where usable Docker image tags are known.
- [ ] **DOCKER-02**: `docker/Dockerfile` supports parameterized ROCm base image selection while preserving the current ROCm 7.1 default behavior.
- [ ] **DOCKER-03**: `scripts/run_docker.sh` supports selecting a declared ROCm target and rejects unknown targets unless an explicit unsafe/untested override is supplied.
- [ ] **DOCKER-04**: Docker preflight checks classify missing `/dev/kfd`, missing `/dev/dri`, unsupported Docker context, and inaccessible GPU devices as `runtime_unavailable` before benchmark execution.
- [ ] **DOCKER-05**: Docker reports record exact requested image repository/tag and, when available, resolved image digest and build arguments.

### uv And PyTorch ROCm

- [ ] **DEPS-01**: Each matrix target records its PyTorch ROCm wheel/index policy, including expected wheel local-version tag and uv index or lock strategy.
- [ ] **DEPS-02**: The default project dependency path remains ROCm 7.1 unless a per-target dependency workflow is explicitly selected and recorded.
- [ ] **DEPS-03**: Missing or unsupported PyTorch ROCm wheels are classified as `pytorch_wheel_unavailable`, not as benchmark failures.
- [ ] **DEPS-04**: CPU, CUDA, wrong-index, or wrong-ROCm PyTorch wheels are detected from installed package metadata and runtime probes.
- [ ] **DEPS-05**: A requested target whose observed PyTorch ROCm wheel, container ROCm user-space, Triton ROCm package, or toolchain version does not match policy is classified as `mixed_version`.
- [ ] **DEPS-06**: Illegal `mixed_version` targets are blocked during preflight before benchmark execution by default.
- [ ] **DEPS-07**: An explicit mixed-version debug override may allow probes or smoke execution to continue, but the resulting entry must remain ineligible for `container_validated`, `host_validated`, score authority, paper-parity authority, or leaderboard authority.

### Runtime Evidence

- [ ] **EVID-01**: Compatibility evidence records host-scope ROCm/driver/device-node evidence separately from container-scope ROCm user-space and toolchain evidence.
- [ ] **EVID-02**: Compatibility evidence records Python runtime metadata, including `torch.__version__`, `torch.version.hip`, `torch.version.cuda`, PyTorch device availability, and `triton-rocm` status when installed.
- [ ] **EVID-03**: Compatibility evidence records GPU metadata, including device count, device name, detected `gfx*` architecture, and visible-device environment variables when available.
- [ ] **EVID-04**: The project emits per-target compatibility JSON and an aggregate compatibility matrix JSON with status counts.
- [ ] **EVID-05**: Runtime setup failures, dependency resolution failures, and benchmark correctness/performance results are reported as distinct evidence categories.
- [ ] **EVID-06**: Compatibility sidecars do not mutate canonical trace JSONL, correctness semantics, timing semantics, scoring schemas, benchmark defaults, or benchmark exit semantics.

### Documentation And Validation

- [ ] **DOCS-01**: Documentation explains that Docker rows validate container ROCm user-space on the recorded host driver/devices and do not prove native host ROCm validation.
- [ ] **DOCS-02**: Documentation explains Target/requested values versus observed host/container/Python/GPU evidence, including why Target is required for matrix interpretation.
- [ ] **DOCS-03**: Documentation states that illegal mixed-version rows are blocked by default and may only continue under explicit debug override without clean validation claims.
- [ ] **DOCS-04**: CPU-safe tests cover status classification, reason-code classification, schema serialization, mixed-version blocking, claim flags, and docs wording guardrails.
- [ ] **DOCS-05**: Docker/script tests cover target selection, default behavior preservation, unknown target rejection, and command construction without requiring live ROCm hardware.
- [ ] **DOCS-06**: Live ROCm validation guidance is marker-gated and records the current host ROCm 7.1.x environment as observed evidence without requiring host reinstall for ROCm 7.0.x or 7.2.x.

## Future Requirements

### Native Host Matrix

- **HOST-01**: Native host validation can be run on separate machines or reinstalled hosts for ROCm 7.0.x, 7.1.x, and 7.2.x.
- **HOST-02**: Native host validation can compare direct host results against Docker user-space results for the same target.

### Extended Hardware Coverage

- **HW-01**: CDNA 3 and CDNA 4 compatibility rows can be marked `host_validated` or `container_validated` only when archived real-hardware evidence exists.
- **HW-02**: Matrix reports can aggregate compatibility status by architecture family after multiple hardware targets have evidence.

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
| Arbitrary unbounded ROCm image tags as normal supported rows | The matrix must remain declared, testable, and auditable. |
| CDNA 3/CDNA 4 validation claims without real hardware artifacts | Hardware claims require archived device-specific evidence. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MATRIX-01 | TBD | Pending |
| MATRIX-02 | TBD | Pending |
| MATRIX-03 | TBD | Pending |
| MATRIX-04 | TBD | Pending |
| MATRIX-05 | TBD | Pending |
| MATRIX-06 | TBD | Pending |
| DOCKER-01 | TBD | Pending |
| DOCKER-02 | TBD | Pending |
| DOCKER-03 | TBD | Pending |
| DOCKER-04 | TBD | Pending |
| DOCKER-05 | TBD | Pending |
| DEPS-01 | TBD | Pending |
| DEPS-02 | TBD | Pending |
| DEPS-03 | TBD | Pending |
| DEPS-04 | TBD | Pending |
| DEPS-05 | TBD | Pending |
| DEPS-06 | TBD | Pending |
| DEPS-07 | TBD | Pending |
| EVID-01 | TBD | Pending |
| EVID-02 | TBD | Pending |
| EVID-03 | TBD | Pending |
| EVID-04 | TBD | Pending |
| EVID-05 | TBD | Pending |
| EVID-06 | TBD | Pending |
| DOCS-01 | TBD | Pending |
| DOCS-02 | TBD | Pending |
| DOCS-03 | TBD | Pending |
| DOCS-04 | TBD | Pending |
| DOCS-05 | TBD | Pending |
| DOCS-06 | TBD | Pending |

**Coverage:**
- v1 requirements: 30 total
- Mapped to phases: 0
- Unmapped: 30

---
*Requirements defined: 2026-05-28*
*Last updated: 2026-05-28 after v1.18 milestone research and mixed-version policy clarification*
