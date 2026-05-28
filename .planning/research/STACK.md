# v1.18 Stack Research: ROCm Version Matrix via Docker

**Project:** SOL ExecBench ROCm Port  
**Researched:** 2026-05-28  
**Scope:** Stack additions and changes only for Docker-selectable ROCm versions, uv/PyTorch ROCm wheel selection, runtime compatibility evidence, and claim guardrails.  
**Overall confidence:** MEDIUM-HIGH. Local integration points are clear. Exact PyTorch ROCm wheel availability must be verified during implementation because ROCm-version wheel indexes change independently from this repo.

## Stack Additions

| Addition | Recommended Shape | Why |
| --- | --- | --- |
| Docker ROCm version selector | Add build args/env vars such as `ROCM_DOCKER_IMAGE=rocm/dev-ubuntu-24.04`, `ROCM_DOCKER_TAG=7.1.1-complete`, and logical `ROCM_VERSION=7.1` in `scripts/run_docker.sh` and `docker/Dockerfile`. | The current Dockerfile hard-codes `rocm/dev-ubuntu-24.04:7.1.1-complete`. Docker Hub currently publishes AMD `rocm/dev-ubuntu-24.04` tags across the desired family, including 7.0.2, 7.1, 7.1-complete, and 7.2.x complete tags. |
| Version matrix manifest | Add a small checked-in manifest, preferably `docker/rocm-version-matrix.toml`, with supported logical rows: `7.0`, `7.1`, `7.2`; Docker tag; PyTorch wheel index name/url; expected torch/torchvision local version suffix; evidence status defaults. | Keeps policy out of shell conditionals and gives roadmap/tests/docs one source of truth. TOML is already natural for this repo because `pyproject.toml` is the central package config; no new parser is needed if the shell only consumes simple exported values or implementation uses Python 3.12 `tomllib`. |
| uv ROCm wheel index rows | Add named uv indexes for the supported ROCm wheel families, for example `pytorch-rocm70`, `pytorch-rocm71`, and `pytorch-rocm72`, all `explicit = true`, with `tool.uv.sources` selecting the active row. | uv official docs support explicit indexes and package-to-index pinning through `tool.uv.sources`. This is the correct guard against accidentally resolving `torch` or `torchvision` from PyPI or the wrong PyTorch index. |
| Lockfile strategy helper | Add a script such as `scripts/lock_rocm_matrix.py` or `scripts/check_rocm_wheels.py` that verifies each matrix row with `uv lock`/metadata checks and emits unavailable rows as evidence instead of silently falling back. | A single `uv.lock` cannot represent multiple mutually exclusive local-version pins selected only by an external Docker build arg unless the project encodes extras/groups or rewrites sources. The safer v1.18 stack is explicit: one default lock remains ROCm 7.1, and matrix rows are generated/verified in containers with recorded lock/check output. |
| Compatibility evidence model | Add new Pydantic models beside `src/sol_execbench/core/environment.py`, e.g. `compatibility_matrix.py`, with schema `sol_execbench.rocm_compatibility_matrix.v1`. | Existing environment evidence already records tools, PyTorch ROCm summary, GPU summaries, and doctor checks. v1.18 needs a higher-level compatibility report that compares host, container user-space, PyTorch wheel, and runtime availability. |
| Container marker env vars | Pass `SOL_EXECBENCH_CONTAINER_ROCM_VERSION`, `SOL_EXECBENCH_CONTAINER_IMAGE`, `SOL_EXECBENCH_CONTAINER_TAG`, and `SOL_EXECBENCH_DOCKER_MATRIX_ROW` into Docker runs. | Runtime evidence needs to distinguish host ROCm from container ROCm user-space. Env vars are low-risk and do not require Docker socket access from inside the container. |
| Evidence probes | Extend existing probes with `hipcc --version`, `rocminfo`, `rocm_agent_enumerator`, optional `rocprofv3 --version`, `torch.__version__`, `torch.version.hip`, `torch.cuda.is_available()`, `torch.cuda.device_count()`, device name, `gcnArchName`, and container marker env vars. | These are already in or adjacent to the current runtime evidence/doctor stack. The missing piece is comparison and classification, not a new profiling or benchmarking dependency. |
| Report command | Add a CLI/report surface such as `sol-execbench rocm-matrix --json` or extend `doctor --json` with a separate matrix payload. | The report must be easy to archive in Docker runs and easy to test without changing canonical trace JSONL. Prefer a standalone command if roadmap wants a user-facing matrix artifact. |

## Existing Files To Modify

| File/Area | Current State | v1.18 Change |
| --- | --- | --- |
| `docker/Dockerfile` | `FROM rocm/dev-ubuntu-24.04:7.1.1-complete AS base`; copies uv `0.5.11`; installs with `uv sync --frozen --no-install-project --all-groups`. | Parameterize `FROM` via pre-`FROM` `ARG ROCM_DOCKER_IMAGE` and `ARG ROCM_DOCKER_TAG`. Record labels/env for selected image/tag. Keep `complete` images for dev/tooling because tests need `hipcc`, `rocprofv3`, `rocminfo`, and ROCm libraries. |
| `scripts/run_docker.sh` | Supports `IMAGE_NAME`, `IMAGE_TAG`, `--build`, Docker run args, and ROCm device passthrough. | Add `ROCM_VERSION` or `ROCM_DOCKER_TAG` selection. During `--build`, pass Docker build args. During `docker run`, pass container marker env vars. Preserve host device checks and Docker Desktop rejection. |
| `pyproject.toml` | Hard-codes `torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, `triton-rocm==3.6.0`, and `pytorch-rocm71`. | Keep ROCm 7.1 as the default project lock. Add named explicit PyTorch ROCm indexes only if implementation can select them deterministically. Do not let generic dependencies resolve from PyTorch indexes. |
| `uv.lock` | Represents the current default ROCm 7.1 environment. | Treat as the canonical default lock, not the whole matrix. If per-version locks are needed, add generated artifacts like `uv-rocm7.0.lock`, `uv-rocm7.1.lock`, `uv-rocm7.2.lock` only after confirming uv workflow and maintenance cost. Do not hand-edit lockfiles. |
| `src/sol_execbench/core/environment.py` | Has `EnvironmentSnapshot`, `PytorchRocmSummary`, `EnvironmentDiagnostics`, `collect_environment_snapshot()`, and `doctor --json` integration. | Extend or compose this evidence. Add fields for host/container distinction only if schema compatibility is preserved; otherwise add a separate `rocm_compatibility_matrix.v1` sidecar/report to avoid mutating existing evidence meaning. |
| `src/sol_execbench/cli/main.py` | Exposes `doctor`, `contract`, main benchmark CLI, env sidecar writing, profiling/static evidence options. | Add matrix report command or optional sidecar writer. Keep canonical trace JSONL untouched. |
| `tests/docker/dependencies/` | Hard readiness checks for PyTorch ROCm, HIP, Triton ROCm, ROCm libraries, and runtime in one container image. | Add version-aware assertions that print selected container image/tag, torch ROCm suffix, `torch.version.hip`, visible device state, and classification. Keep failures hard for selected rows that claim runnable status. |
| `docs/CONFIGURATION.md`, `docs/TESTING.md`, `docs/rocm.md`, `docs/CLAIMS.md` | Document single-image Docker usage and ROCm 7.1 wheel resolution. | Document matrix variables, supported rows, claim boundary language, and examples for building/running each selected user-space container. |

## Version/Dependency Strategy

### Recommended Version Variables

Use these names consistently:

| Variable | Owner | Example | Meaning |
| --- | --- | --- | --- |
| `ROCM_VERSION` | User-facing wrapper input | `7.1` | Logical matrix row. Should map to a Docker tag and wheel policy. |
| `ROCM_DOCKER_IMAGE` | Docker wrapper/build | `rocm/dev-ubuntu-24.04` | Base image repository. |
| `ROCM_DOCKER_TAG` | Docker wrapper/build | `7.1.1-complete` | Exact base image tag. |
| `PYTORCH_ROCM_INDEX` | Matrix metadata / uv helper | `https://download.pytorch.org/whl/rocm7.1` | Wheel index intended for torch-family packages. |
| `PYTORCH_ROCM_SUFFIX` | Evidence checks | `rocm7.1` | Expected local version suffix in `torch.__version__`/`torchvision.__version__`. |
| `SOL_EXECBENCH_CONTAINER_ROCM_VERSION` | Runtime evidence | `7.1` | Logical row captured in evidence. |
| `SOL_EXECBENCH_CONTAINER_IMAGE` | Runtime evidence | `rocm/dev-ubuntu-24.04:7.1.1-complete` | Exact selected container user-space image. |

### Docker Image Base Strategy

Use `rocm/dev-ubuntu-24.04:*complete` images for v1.18 rows. The current stack needs a development image, not only a runtime image, because native HIP/C++ extension builds use `hipcc` through `torch.utils.cpp_extension`, dependency checks exercise ROCm libraries, and optional evidence uses toolchain/profiler commands.

Recommended initial manifest rows:

| Logical ROCm | Docker Tag | Default Claim State | Notes |
| --- | --- | --- | --- |
| `7.0` | `7.0.2-complete` | `not_tested` until run | Published AMD Docker tag exists. Validate with host driver before claiming container compatibility. |
| `7.1` | `7.1.1-complete` or current exact 7.1.x complete tag | `container_validated` only after run | Current repo default is effectively 7.1. This remains the default lock and default Docker image. |
| `7.2` | Latest pinned `7.2.x-complete`, not floating `latest` | `not_tested` until run | Pin exact tag, for example `7.2.3-complete` if selected, and avoid `latest` in evidence-producing runs. |

Do not use floating Docker tags for validation evidence. If a row uses `7.2-complete` or `latest` for convenience, reports must classify it as non-reproducible unless the resolved digest is recorded.

### uv/PyTorch ROCm Wheel Strategy

Keep `pyproject.toml`'s ROCm 7.1 dependency set as the default project environment until the milestone explicitly proves multi-lock or generated-lock workflow. For matrix rows, use a checked manifest and helper command that:

1. selects the target PyTorch ROCm index;
2. verifies `torch`, `torchvision`, and `triton-rocm` availability for Python `>=3.12,<3.14`;
3. records the resolved versions and local suffixes;
4. fails or classifies `pytorch_wheel_unavailable` instead of falling back to CPU, CUDA, PyPI, or the wrong ROCm index.

Do not rely on uv index search order alone. uv defaults to first-index behavior and supports explicit package-to-index sources; use explicit PyTorch indexes for torch-family packages and PyPI as the default for everything else.

`triton-rocm==3.6.0` currently resolves from the PyTorch ROCm root index. Keep it pinned until tests prove a per-ROCm-version Triton mapping is required. Treat Triton failures separately from `torch` wheel availability because a PyTorch ROCm wheel can import while Triton ROCm kernels still fail.

### Compatibility Status Vocabulary

Use the milestone vocabulary directly:

| Status | Meaning |
| --- | --- |
| `host_validated` | Native host ROCm stack was tested directly on this host/user-space combination. Docker evidence alone must not produce this status. |
| `container_validated` | Selected Docker user-space image ran the required checks on the host driver/devices. |
| `mixed_version` | Host driver/runtime evidence, container user-space, PyTorch HIP build tag, or expected matrix row disagree. |
| `pytorch_wheel_unavailable` | Required torch-family ROCm wheel was not available or did not match the selected matrix row. |
| `runtime_unavailable` | Container built or dependencies installed, but PyTorch ROCm/HIP runtime/device access failed. |
| `not_tested` | Matrix row exists but no evidence artifact has been produced. |

## Integration Boundaries

| Boundary | Rule |
| --- | --- |
| Docker user-space vs host driver | Docker rows validate container ROCm user-space on the current host kernel/driver/devices. They do not prove that ROCm 7.0.x, 7.1.x, or 7.2.x is natively installed or validated on the host. |
| PyTorch ROCm wheel vs ROCm container | `torch.version.hip` and the `+rocmX.Y` local version suffix are PyTorch build metadata. They must be compared with, but not assumed identical to, `/opt/rocm` user-space in the container. |
| Evidence vs benchmark authority | Matrix evidence is diagnostic compatibility evidence. It must not change correctness, timing, scoring, paper parity, static evidence, profiling, or leaderboard semantics. |
| Host visibility | `rocminfo`, `/dev/kfd`, `/dev/dri`, and PyTorch device visibility remain runtime gates. A container image build success is not runtime validation. |
| Existing environment snapshot | Preserve existing `sol_execbench.environment_snapshot.v1` semantics. Prefer composing it into a new compatibility report instead of overloading existing fields. |
| CUDA terminology | PyTorch ROCm still exposes AMD GPUs via `torch.cuda` APIs. Guardrails should allow that compatibility namespace while rejecting CUDA/NVIDIA runtime claims. |

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| PyTorch ROCm wheel gaps across 7.0/7.2 | Matrix row cannot install or silently installs wrong backend. | Add wheel availability checks and classify `pytorch_wheel_unavailable`; never fall back silently. |
| Single `uv.lock` overclaims matrix reproducibility | Default lock is mistaken for all ROCm versions. | State that `uv.lock` is default ROCm 7.1. Generate per-row lock/check artifacts only through a scripted workflow. |
| Host/container version mismatch | Results are reported as native validation when only container user-space was tested. | Record host evidence, container image/tag, PyTorch HIP build, and status separately; reserve `host_validated` for native runs only. |
| Floating Docker tags drift | Re-running the same command later validates a different stack. | Pin exact Docker tags and, for release evidence, record resolved image digest. |
| ROCm base image lacks required developer tools | HIP/C++ extension builds or evidence probes fail. | Use `complete` dev images for matrix rows; keep runtime-only images out of v1.18. |
| Triton ROCm compatibility differs from PyTorch import compatibility | PyTorch smoke passes while Triton examples fail. | Keep Triton checks as separate evidence and do not collapse them into PyTorch wheel availability. |
| Docker Desktop or missing device passthrough | False runtime failures on unsupported Docker context. | Keep existing `scripts/run_docker.sh` native Linux Docker checks and `/dev/kfd`/`/dev/dri` validation. |
| Expanding dependency scope | Milestone becomes a general environment manager. | Do not add conda, Spack, Nix, host ROCm installers, kernel driver management, or alternate benchmark backends. |

## What Not To Add

- Do not add host ROCm installers or scripts that reinstall the system ROCm stack.
- Do not add a CUDA/NVIDIA backend or CUDA compatibility matrix.
- Do not add conda/mamba as a second package manager.
- Do not add Docker-in-Docker or require access to the Docker socket inside benchmark containers.
- Do not make matrix evidence mandatory for normal `sol-execbench` runs.
- Do not treat Docker container validation as native host ROCm validation.
- Do not switch to floating Docker image tags for release evidence.
- Do not hand-edit `uv.lock`; use uv commands/scripts only.

## Sources

- Local: `.planning/PROJECT.md`, `.planning/STATE.md`, `.planning/MILESTONES.md`, `pyproject.toml`, `docker/Dockerfile`, `docker/entrypoint.sh`, `scripts/run_docker.sh`, `src/sol_execbench/core/environment.py`.
- AMD Docker Hub `rocm/dev-ubuntu-24.04` tags, crawled 2026-05-28: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags
- PyTorch previous versions and ROCm wheel-index pattern, crawled 2026-05-28: https://pytorch.org/get-started/previous-versions/
- AMD ROCm PyTorch Docker install example for ROCm 7.2/PyTorch 2.9.1, crawled 2026-05-28: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/native_linux/install-pytorch.html
- uv package index docs, explicit indexes, and index strategy, crawled 2026-05-28: https://docs.astral.sh/uv/concepts/indexes/
- uv dependency source docs, package-to-index pinning, crawled 2026-05-28: https://docs.astral.sh/uv/concepts/projects/dependencies/
