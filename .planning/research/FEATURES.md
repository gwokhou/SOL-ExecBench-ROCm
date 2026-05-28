# Feature Landscape

**Domain:** Docker-based ROCm version compatibility matrix for SOL ExecBench ROCm Port  
**Milestone:** v1.18 ROCm Version Matrix via Docker  
**Researched:** 2026-05-28  
**Overall confidence:** HIGH for repo-local feature surface and claim boundaries; MEDIUM for exact PyTorch/ROCm wheel availability by version because wheel publication is external and must be probed at runtime.

## Feature Categories

### 1. Matrix Definition And Selection

v1.18 should expose a small, explicit ROCm matrix instead of implying that one Docker image validates every ROCm stack. Users need to select a target ROCm user-space version, run the benchmark diagnostics inside that container, and get a durable status record.

Expected matrix axes:

| Axis | Required Values | Notes |
| --- | --- | --- |
| ROCm container user-space | `7.0.x`, `7.1.x`, `7.2.x` entries when image tags are configured | The project currently defaults to ROCm 7.1.1; the matrix should make version choice explicit. |
| Host ROCm/kernel driver evidence | detected version or `unknown` | Host driver/runtime evidence is context, not the selected container stack. |
| PyTorch ROCm wheel tag | `rocm7.0`, `rocm7.1`, `rocm7.2`, or unavailable reason | Must be recorded separately from container ROCm version. |
| GPU architecture | `gfx1200`, `gfx94*`, other detected `gfx*`, or `unknown` | Do not upgrade hardware validation claims without archived hardware evidence. |
| Validation mode | `container_user_space`, `native_host`, or `mixed` | v1.18 focuses on container user-space validation, not native host reinstall validation. |

Recommended user-visible entry points:

```bash
# Build and run one selected ROCm user-space image.
ROCM_VERSION=7.1.1 ./scripts/run_docker.sh --build

# Emit a compatibility report without running a benchmark problem.
uv run sol-execbench compatibility --json

# Run the configured Docker matrix and write sidecars/reports.
uv run scripts/run_rocm_matrix.py --output out/rocm-matrix
```

The exact command names can be refined during planning, but the feature surface should include both a single-environment probe and a multi-version matrix runner.

### 2. Runtime Evidence Sidecars

The matrix must produce machine-readable compatibility evidence. It should extend the existing sidecar pattern used by environment, profiling, toolchain routing, and static evidence artifacts. It must not mutate canonical trace JSONL, correctness, timing, scoring, or baseline comparison behavior.

Recommended sidecar:

```text
out/rocm-matrix/compatibility-matrix.json
out/rocm-matrix/rocm-7.1.1/compatibility.json
out/rocm-matrix/rocm-7.1.1/doctor.json
out/rocm-matrix/rocm-7.1.1/uv-lock-or-sync.txt
```

Recommended schema identity:

```text
sol_execbench.rocm_compatibility_matrix.v1
sol_execbench.rocm_compatibility_entry.v1
```

Required machine-readable fields:

| Field | Required Behavior |
| --- | --- |
| `schema_version` | Stable version string for automation. |
| `generated_at` | UTC timestamp. |
| `matrix_entry_id` | Stable ID such as `rocm-7.1.1-pytorch-rocm7.1-gfx1200`. |
| `requested_rocm_version` | Requested container ROCm version or native host version. |
| `container_image` | Image repository, tag, digest if available, Dockerfile path, build args. |
| `host_rocm` | Host ROCm evidence from available probes, or `unknown`. |
| `container_rocm` | Container ROCm evidence from `hipcc`, `rocminfo`, `/opt/rocm`, or package metadata. |
| `pytorch` | `torch.__version__`, `torch.version.hip`, `torch.version.cuda`, wheel/index source, device availability. |
| `triton_rocm` | Installed version or unavailable reason. |
| `toolchain` | `hipcc`, `rocminfo`, `rocm_agent_enumerator`, `amd-smi`/`rocm-smi`, and relevant LLVM tool versions when available. |
| `gpu` | Device name, `gcnArchName`, visible-device environment variables, and device count. |
| `uv` | Lock/sync strategy, selected PyTorch ROCm index, and dependency resolution status. |
| `status` | One of the v1.18 compatibility states below. |
| `reason_codes` | Stable reason codes for automation. |
| `claim_boundary` | Explicit booleans distinguishing container validation from native host validation and hardware validation. |
| `artifacts` | Paths to logs, doctor JSON, command transcripts, and optional smoke trace artifacts. |

### 3. Compatibility States

The report must use a bounded vocabulary so requirements and tests can assert behavior.

| State | Meaning | Testable Trigger |
| --- | --- | --- |
| `host_validated` | Native host ROCm stack was validated directly without relying on Docker user-space substitution. | A native run records host ROCm, PyTorch ROCm, GPU visibility, and smoke behavior from the host environment. Not a default Docker matrix outcome. |
| `container_validated` | A Docker container with the requested ROCm user-space stack passed required probes and optional smoke benchmark checks. | Container `torch.version.hip` is set, `torch.version.cuda` is `None`, device is visible, ROCm tools report usable status, and smoke checks pass. |
| `mixed_version` | Host, container, PyTorch wheel, Triton ROCm, or HIP toolchain versions are materially different in a way that affects claim wording. | Example: container ROCm 7.2 with PyTorch `+rocm7.1`, or host ROCm evidence conflicts with container user-space claim. |
| `pytorch_wheel_unavailable` | The requested ROCm/Python/PyTorch combination cannot be resolved from configured indexes. | `uv lock`/`uv sync` or a probe command reports no matching `torch`/`torchvision` ROCm wheel. |
| `runtime_unavailable` | Required ROCm runtime/device access is missing. | Missing `/dev/kfd`, missing `/dev/dri`, Docker Desktop context, `torch.cuda.is_available() == False`, no AMD device, or `rocminfo` cannot enumerate a GPU. |
| `not_tested` | Matrix entry is declared but no validation attempt was made. | Entry exists in configuration but has no command transcript, probe payload, or result artifact. |

Recommended reason codes:

| Reason Code | Applies To | Meaning |
| --- | --- | --- |
| `docker_desktop_context` | `runtime_unavailable` | Docker context cannot pass ROCm devices through. |
| `missing_kfd` | `runtime_unavailable` | `/dev/kfd` missing or inaccessible. |
| `missing_dri` | `runtime_unavailable` | `/dev/dri` missing or inaccessible. |
| `torch_hip_unset` | `runtime_unavailable`, `mixed_version` | PyTorch is not a ROCm build. |
| `torch_cuda_build_detected` | `mixed_version` | CUDA wheel or CUDA metadata is present in a ROCm matrix entry. |
| `torch_device_unavailable` | `runtime_unavailable` | PyTorch ROCm build imports but cannot see a GPU. |
| `pytorch_index_missing_wheel` | `pytorch_wheel_unavailable` | Configured PyTorch ROCm index has no matching wheel. |
| `pytorch_python_incompatible` | `pytorch_wheel_unavailable` | Python version has no compatible wheel for the requested stack. |
| `container_rocm_mismatch` | `mixed_version` | Requested and detected container ROCm versions differ. |
| `torch_rocm_mismatch` | `mixed_version` | PyTorch ROCm build tag differs from container ROCm policy. |
| `triton_rocm_mismatch` | `mixed_version` | Triton ROCm package does not match the requested runtime policy. |
| `host_container_claim_boundary` | `mixed_version`, `container_validated` | Host and container versions differ; report must not claim native host validation. |
| `entry_configured_not_run` | `not_tested` | Matrix entry is present but lacks execution evidence. |

### 4. Claim-Boundary Behavior

Every report must say what was validated and what was not. The key product behavior is preventing a container user-space pass from being overstated as a native host ROCm pass.

Required claim flags:

| Flag | Required Value/Behavior |
| --- | --- |
| `container_user_space_validated` | `true` only when the selected Docker entry passed required probes. |
| `native_host_validated` | `true` only for a direct host validation run. Docker matrix entries should normally set this to `false`. |
| `hardware_validated` | `true` only when real device evidence for that architecture is archived for the entry. |
| `paper_parity_authority` | Always `false` for v1.18 compatibility sidecars. |
| `score_authority` | Always `false`. |
| `leaderboard_authority` | Always `false`. |
| `diagnostic_compatibility_evidence` | Always `true` for these sidecars. |

Required wording in Markdown reports:

- Say "container ROCm user-space validated" for passing Docker entries.
- Say "native host ROCm validated" only for direct host runs.
- Say "mixed version" when container ROCm, PyTorch ROCm wheel tag, Triton ROCm package, or host evidence do not align.
- Say "unavailable" rather than "failed benchmark" when the matrix cannot obtain runtime or wheel prerequisites.

## Table Stakes

| Feature | Why Expected | Acceptance Behavior |
| --- | --- | --- |
| Select ROCm Docker user-space version | v1.18 exists to test multiple ROCm user-space versions without host reinstall. | Docker build/run accepts a configured ROCm version/tag and records requested vs detected container ROCm version. |
| Preserve existing default Docker behavior | Current users rely on `./scripts/run_docker.sh --build`. | Existing command still works and defaults to the documented current ROCm image unless a version is explicitly selected. |
| Refuse unsupported Docker contexts | Existing wrapper already rejects Docker Desktop; matrix must retain that guardrail. | Docker Desktop context yields `runtime_unavailable` with `docker_desktop_context`, not a misleading matrix pass. |
| Check ROCm device passthrough | ROCm containers need `/dev/kfd` and `/dev/dri`. | Missing devices produce `runtime_unavailable` with stable reason codes. |
| Probe container ROCm runtime | Users need proof of the container user-space stack. | Report records `hipcc --version`, `rocminfo`/agent availability, and `/opt/rocm` evidence when available. |
| Probe PyTorch ROCm identity | PyTorch wheel selection is central to v1.18. | Report records `torch.__version__`, `torch.version.hip`, `torch.version.cuda`, `torch.cuda.is_available()`, device name, and `gcnArchName`. |
| Detect CUDA or CPU PyTorch wheels | Wrong wheels silently invalidate compatibility claims. | CUDA/CPU wheels produce `mixed_version` or `runtime_unavailable` with `torch_hip_unset`/`torch_cuda_build_detected`. |
| Coordinate `uv` indexes with matrix entry | Current `pyproject.toml` pins ROCm 7.1 PyTorch indexes. Matrix entries need explicit wheel policy. | Each entry records the selected PyTorch ROCm index/source and whether resolution succeeded. |
| Represent missing PyTorch wheels distinctly | Wheel absence is different from GPU runtime absence. | No matching wheel yields `pytorch_wheel_unavailable`, not `runtime_unavailable`. |
| Detect mixed container/wheel versions | ROCm containers and PyTorch wheels can intentionally or accidentally differ. | ROCm 7.2 container with `+rocm7.1` PyTorch reports `mixed_version` unless policy explicitly allows it with a warning. |
| Emit compatibility matrix JSON | Downstream requirements need machine-readable acceptance. | `compatibility-matrix.json` contains one entry per configured target and aggregate counts by state. |
| Emit per-entry sidecar JSON | Researchers need full evidence for each environment. | Every attempted entry writes `compatibility.json` with probes, artifacts, status, and claim flags. |
| Emit concise Markdown or Rich report | Humans need to understand matrix results quickly. | Report table shows requested ROCm, image tag, PyTorch ROCm tag, GPU arch, state, and primary reason. |
| Support `not_tested` entries | Roadmaps often declare a matrix before all entries are run. | Configured-but-unrun entries appear as `not_tested` with `entry_configured_not_run`. |
| Keep canonical trace JSONL unchanged | Compatibility evidence is diagnostic, not benchmark output. | Tests verify matrix sidecars do not add fields to trace records. |
| Do not alter correctness/timing/scoring | Compatibility status must not become score authority. | Smoke benchmark failure can be recorded in compatibility artifacts but does not change score schema or baseline command semantics. |
| Reuse `sol-execbench doctor --json` | Existing runtime diagnostics are the right integration point. | Matrix stores doctor JSON per entry instead of duplicating every diagnostic rule. |
| Optional smoke benchmark hook | A runtime import probe is useful but weaker than a minimal actual run. | Matrix can run a bounded sample problem and record pass/fail/unavailable separately from compatibility probes. |
| Stable reason code taxonomy | Requirements and docs need testable outcomes. | Unit tests cover each required state and representative reason code. |
| Documentation updates | Users need exact claim wording. | README/docs explain container vs host validation, uv/PyTorch coordination, and state meanings. |
| CPU-safe tests | CI may lack ROCm hardware. | Tests use injected command/probe fixtures for matrix classification and sidecar serialization. |

## Differentiators

| Feature | Value Proposition | Acceptance Behavior |
| --- | --- | --- |
| Image digest capture | Makes a passing container entry reproducible after tags move. | Report records image digest when Docker exposes it; otherwise `unknown` with warning. |
| Matrix config file | Lets maintainers add/drop ROCm versions without editing scripts. | A checked-in config declares version, image, PyTorch index, expected wheel tag, and optional smoke command. |
| Native-host comparison row | Clarifies how container results relate to the host stack. | Optional host probe emits `host_validated` or host-side unavailable state without requiring Docker. |
| JSON Schema export | Enables external CI/report consumers to validate sidecars. | Schema file or CLI schema output validates example reports. |
| GitHub Actions artifact mode | Useful for non-GPU CI to publish planned/not-tested matrix. | CI can generate `not_tested` or fixture-based reports without claiming validation. |
| Wheel availability preflight | Saves time before Docker builds. | A command checks configured PyTorch ROCm wheel availability and reports `pytorch_wheel_unavailable` before container execution. |
| Matrix diff report | Helps compare v1.18 runs over time. | Tool compares two matrix JSON files and highlights state/version changes. |
| Compatibility badge data | Makes docs/release notes easy to update from JSON. | Report emits a compact summary consumable by docs generation. |
| Per-architecture aggregation | Shows RDNA 4 vs CDNA 3/CDNA 4 evidence boundaries. | Matrix report groups entries by detected `gfx*` and never infers untested architectures. |
| Command transcript capture | Improves auditability. | Each entry stores bounded stdout/stderr tails and exact commands for Docker, uv, and probes. |

## Out Of Scope

| Out Of Scope Item | Why Avoid In v1.18 | Instead |
| --- | --- | --- |
| Native host ROCm reinstall automation | The milestone is Docker-based matrix coverage. Host package management is risky and environment-specific. | Allow an optional host probe that records `host_validated` only when run directly. |
| Claiming Docker pass as native host validation | Container user-space validation does not prove the host ROCm install behaves the same. | Use explicit claim flags and report wording. |
| Full paper dataset validation across every ROCm version | This would expand v1.18 into paper-scale benchmarking. | Use bounded smoke checks and compatibility probes. |
| Score or leaderboard eligibility changes | Version compatibility is diagnostic/reproducibility evidence. | Keep scoring and leaderboard claims unchanged. |
| CUDA/NVIDIA compatibility matrix | Project is ROCm-only. | Detect CUDA wheels as invalid/mixed for ROCm entries. |
| Supporting arbitrary ROCm versions by default | A broad open-ended matrix becomes untestable. | Start with declared 7.0.x, 7.1.x, 7.2.x entries. |
| Guaranteeing PyTorch wheels exist for every ROCm version | Wheel publication is external to the project. | Probe and report `pytorch_wheel_unavailable`. |
| Hiding mixed versions behind "compatible" | Mixed stacks may be useful but must not be overclaimed. | Report `mixed_version` with explicit policy/reason. |
| Replacing environment sidecars | v1.13 environment evidence remains useful. | Reference or embed existing doctor/environment outputs in compatibility sidecars. |
| Making Docker mandatory for normal benchmark runs | Existing native and single-container workflows should remain. | Matrix tooling is additive. |
| Full CDNA 3/CDNA 4 hardware validation | Hardware validation requires real archived runs on those devices. | Record architecture-specific status only when direct evidence exists. |

## Acceptance Signals

### User-Visible Acceptance

| Scenario | Expected Result |
| --- | --- |
| User runs current Docker command with no version override | It builds/runs the default image exactly as before and documentation identifies the default ROCm stack. |
| User selects ROCm 7.1.1 container | Docker build/run uses the configured 7.1.1 image tag and report records `requested_rocm_version=7.1.1`. |
| User selects a configured ROCm 7.2.x entry with PyTorch ROCm 7.1 wheel policy | Report returns `mixed_version` unless policy explicitly marks that combination as accepted-with-warning. |
| User runs matrix without `/dev/kfd` | Entry returns `runtime_unavailable` with `missing_kfd`; command exits in a documented nonzero or report-only mode. |
| User runs matrix where PyTorch wheel cannot resolve | Entry returns `pytorch_wheel_unavailable`; report includes uv/index evidence. |
| User asks for Markdown summary | Summary table includes all entries and uses only the approved states. |
| User reads docs | Docs clearly distinguish `container_validated` from `host_validated`. |

### Machine-Readable Acceptance

| Capability | Required Assertion |
| --- | --- |
| Schema identity | JSON contains `schema_version=sol_execbench.rocm_compatibility_matrix.v1`. |
| State enum | Parser rejects unknown states and accepts `host_validated`, `container_validated`, `mixed_version`, `pytorch_wheel_unavailable`, `runtime_unavailable`, `not_tested`. |
| Claim flags | Docker entries cannot set `native_host_validated=true` unless validation mode is explicitly native host. |
| Aggregate counts | Matrix JSON reports counts for every state, including zero counts. |
| Entry artifacts | Every attempted entry has command/probe artifact references or explicit unavailable reasons. |
| PyTorch metadata | Every attempted entry records `torch.version.hip`, `torch.version.cuda`, and device availability when Python imports succeed. |
| uv metadata | Every attempted entry records index/source policy and resolution status. |
| Mixed-version classifier | Fixture tests cover container/PyTorch mismatch, host/container mismatch, and CUDA wheel detection. |
| Runtime-unavailable classifier | Fixture tests cover Docker Desktop, missing `/dev/kfd`, missing `/dev/dri`, and no PyTorch device. |
| Wheel-unavailable classifier | Fixture tests cover uv no-solution/no-wheel output and Python-version incompatibility. |
| Not-tested classifier | Declared entries without execution artifacts serialize as `not_tested`. |
| Trace isolation | Tests prove matrix commands do not mutate canonical trace JSONL schemas. |
| Documentation guardrails | Tests or static checks catch wording that equates container validation with native host validation. |

### Minimum MVP For v1.18

1. Version-selectable Docker build/run for declared ROCm image tags.
2. Compatibility entry sidecar with probes for Docker/runtime/PyTorch/uv/GPU metadata.
3. Matrix aggregate JSON and human-readable report.
4. Exact state vocabulary: `host_validated`, `container_validated`, `mixed_version`, `pytorch_wheel_unavailable`, `runtime_unavailable`, `not_tested`.
5. Claim-boundary flags and docs that prevent container validation from being overstated.
6. CPU-safe fixture tests for classification, serialization, and report rendering.

## Sources

- Repo: `.planning/PROJECT.md`, `.planning/STATE.md`, `.planning/MILESTONES.md`, `README.md`, `docs/rocm.md`, `docs/CLAIMS.md`, `docs/CONFIGURATION.md`, `docs/GETTING-STARTED.md`, `scripts/run_docker.sh`, `pyproject.toml`.
- Official PyTorch previous versions page lists ROCm wheel index commands including `torch==2.10.0` with `--index-url https://download.pytorch.org/whl/rocm7.1`: https://pytorch.org/get-started/previous-versions/
- AMD ROCm PyTorch installation docs list validated ROCm/PyTorch Docker image inventories for ROCm 7.1.1: https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.1/install/3rd-party/pytorch-install.html
- Docker Hub exposes ROCm `dev-ubuntu-24.04` tags such as `7.1.1-complete` and `7.2.1-complete`: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags
- uv documentation recommends explicit PyTorch indexes/sources so PyTorch packages come from the intended index while generic packages remain on PyPI: https://docs.astral.sh/uv/guides/integration/pytorch/
