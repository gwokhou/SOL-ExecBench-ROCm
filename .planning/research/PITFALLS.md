# v1.18 Pitfalls: ROCm Version Matrix via Docker

**Project:** SOL ExecBench ROCm Port
**Milestone:** v1.18 ROCm Version Matrix via Docker
**Researched:** 2026-05-28
**Question:** What common mistakes and guardrails are needed when adding Docker-based ROCm version matrix validation and uv/PyTorch wheel selection?

## Research Basis

Repo evidence reviewed:

- `pyproject.toml` currently pins Linux/Windows `torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, `triton-rocm==3.6.0`, and explicit uv indexes for PyTorch ROCm 7.1.
- `docker/Dockerfile` currently hardcodes `rocm/dev-ubuntu-24.04:7.1.1-complete` and installs with `uv sync --frozen`.
- `scripts/run_docker.sh` currently passes `/dev/kfd`, `/dev/dri`, `--group-add video`, `--security-opt seccomp=unconfined`, `--ipc=host`, and rejects Docker Desktop contexts.
- `src/sol_execbench/core/environment.py` currently records ROCm tools, visible-device variables, PyTorch version, `torch.version.hip`, `torch.version.cuda`, device name, and gfx target, but not host/container split or compatibility-matrix state.
- `docs/CLAIMS.md` already enforces claim boundaries for runtime, profiling, toolchain routing, static evidence, AMD-native score evidence, and hardware validation.
- `tests/docker/dependencies/` already contains hard ROCm container checks for PyTorch ROCm, runtime tools, libraries, Triton ROCm, and dependency source configuration.

External evidence reviewed:

- AMD Container Runtime Toolkit states Docker Desktop on Linux and rootless mode are not supported for GPU workloads, host user permissions need `render` and `video` groups, and ROCm/driver mismatches can cause runtime failures or undefined behavior.
- AMD amdgpu 30.20.1 docs state that driver release is compatible with ROCm 7.1.x, 7.0.x, 6.4.x, and 6.3.x.
- AMD ROCm Docker docs state containers share the host kernel driver, require `/dev/kfd` plus `/dev/dri`, and `rocminfo` or `amd-smi` inside the container only enumerates GPUs passed into that container.
- PyTorch previous-version docs list PyTorch 2.10.0 ROCm 7.1 installation through `https://download.pytorch.org/whl/rocm7.1`.
- uv docs state that `[tool.uv.sources]` with an `index` pins a package to that index, `explicit = true` limits index use to explicitly selected packages, and uv's default first-index behavior is designed to avoid dependency-confusion problems.

## Phase Placement

| Phase | Name | Pitfalls Owned |
| --- | --- | --- |
| Phase 1 | Matrix Contract And Status Vocabulary | Compatibility states, claim taxonomy, sidecar/report schema, overclaim prevention. |
| Phase 2 | Docker Image Selection And Build Args | ROCm image tags, build args, host-driver checks, device passthrough, Docker Desktop/rootless guardrails. |
| Phase 3 | uv And PyTorch ROCm Wheel Coordination | PyTorch local version tags, uv indexes/sources, lockfile strategy, unavailable wheel states. |
| Phase 4 | Runtime Evidence And Compatibility Reports | Host/container/PyTorch/Triton/toolchain evidence, mixed-version detection, matrix report generation. |
| Phase 5 | Validation Workflow, Docs, And CI Guardrails | CPU-safe tests, Docker-only checks, docs language, archived validation evidence, CI feasibility. |

## Pitfalls

### 1. Treating Container ROCm As Host ROCm Validation

**What goes wrong:** A run in `rocm/dev-ubuntu-24.04:7.0.x` or `7.1.x` is reported as "ROCm 7.0 host validated" or "native ROCm 7.0 validated" even though the host kernel driver remains the real host driver.

**Why it happens:** Docker changes the ROCm user-space stack, not the host kernel-mode driver boundary. AMD docs are explicit that ROCm containers share the host kernel and require a compatible host driver plus `/dev/kfd` and `/dev/dri`.

**Consequences:** The matrix overstates coverage. A user on a native ROCm 7.0 host may see behavior different from a ROCm 7.0 user-space container running over a ROCm 7.1.x host driver.

**Prevention:**

- Use status names that encode the evidence boundary: `host_validated`, `container_validated`, `mixed_version`, `runtime_unavailable`, `pytorch_wheel_unavailable`, `not_tested`.
- Require matrix rows to record `host_driver_version`, `host_rocm_version` when detectable, `container_image`, `container_rocm_version`, `torch_version`, `torch_rocm_tag`, `torch.version.hip`, `triton_rocm_version`, and `gfx_target`.
- Docs must say "container user-space validated on host ROCm/driver X", not "native ROCm Y validated".

**Required Guardrails:**

- A schema-level field for `validation_scope` with allowed values like `native_host`, `docker_user_space`, and `not_validated`.
- A report warning when `container_rocm_version != host_rocm_version` or when host ROCm cannot be detected.
- Documentation tests that reject "host validated" language for Docker-only evidence.

**Phase Placement:** Phase 1, Phase 4, Phase 5.

### 2. Assuming Any ROCm Container Version Works With The Current Host Driver

**What goes wrong:** The matrix lets users select arbitrary ROCm image tags, then interprets runtime crashes as benchmark failures rather than host-driver/container incompatibility.

**Why it happens:** The current host is ROCm 7.1.x, but target containers include 7.0.x, 7.1.x, and possibly 7.2.x. AMD documents driver/runtime compatibility windows, but mismatches can still fail at runtime depending on driver release, GPU generation, partitioning, and library use.

**Consequences:** Invalid rows pollute benchmark results. The project may report "ROCm 7.2 failed" when the real result is "container user-space not compatible with this host driver".

**Prevention:**

- Add a preflight that runs before benchmark execution: host device nodes, Docker context, rootless/Docker Desktop detection where possible, `rocminfo` inside container, `amd-smi` or `rocm-smi`, PyTorch ROCm import, device memory copy, event timing.
- Treat compatibility preflight failures as matrix states, not benchmark correctness failures.
- Keep a compatibility policy table with the tested host driver/ROCm release and supported container ROCm user-space rows.

**Required Guardrails:**

- `runtime_unavailable` and `mixed_version` states must be first-class.
- Reports must include preflight command status and remediation hints.
- A selected ROCm image tag outside the supported matrix should require explicit `--allow-untested-rocm-image` or equivalent.

**Phase Placement:** Phase 2, Phase 4.

### 3. Hardcoding The Docker Base Image In Too Many Places

**What goes wrong:** `docker/Dockerfile`, docs, tests, and scripts drift: one says ROCm 7.1.1, another builds 7.0, and dependency tests still assert the old PyTorch ROCm index.

**Why it happens:** The current Dockerfile has `FROM rocm/dev-ubuntu-24.04:7.1.1-complete` and current docs repeat that tag. Matrix support needs one source of truth.

**Consequences:** Matrix rows become irreproducible. Users rebuild a tag that does not match the reported row.

**Prevention:**

- Add Docker build args such as `ROCM_IMAGE=rocm/dev-ubuntu-24.04` and `ROCM_IMAGE_TAG=7.1.1-complete`, then derive image labels from those args.
- Make `scripts/run_docker.sh` accept a bounded ROCm selector such as `ROCM_VERSION=7.1.1` or `--rocm-version 7.1.1`.
- Emit Docker image labels for project version, ROCm image tag, PyTorch ROCm target, uv lock strategy, and build timestamp.

**Required Guardrails:**

- Tests must assert Dockerfile supports parameterized base tags instead of a single hardcoded tag.
- Matrix report rows must store the full image digest or at least resolved image tag plus build args.
- Docs must show the matrix command, not only the default image command.

**Phase Placement:** Phase 2, Phase 5.

### 4. Mixing PyTorch ROCm Wheels And Container ROCm User Space

**What goes wrong:** A ROCm 7.0 container installs `torch==2.10.0+rocm7.1`, or a ROCm 7.1 container accidentally resolves a CPU/CUDA wheel. The benchmark still imports `torch`, but it is not validating the intended stack.

**Why it happens:** PyTorch ROCm wheel support is version-specific. The current project pins `+rocm7.1` and routes `torch`/`torchvision` to `https://download.pytorch.org/whl/rocm7.1`. uv will respect the lockfile when `--frozen` is used, so a matrix build cannot magically select a different wheel unless the dependency and lock strategy change deliberately.

**Consequences:** The matrix validates the wrong PyTorch/ROCm pairing. Mixed-version behavior may appear as kernel correctness, timing, Triton, or extension-build failures.

**Prevention:**

- Treat PyTorch wheel selection as part of the matrix contract, not an incidental install detail.
- Require runtime evidence to compare intended `pytorch_rocm_target` with `torch.__version__`, local version tag, `torch.version.hip`, `torch.version.cuda`, and `torch.cuda.is_available()`.
- For every ROCm row, define whether PyTorch wheels are `available`, `unavailable`, or `intentionally reused_with_warning`.

**Required Guardrails:**

- Add `pytorch_wheel_unavailable` and `mixed_version` states.
- Add tests for parsing local version tags such as `2.10.0+rocm7.1`.
- Fail or mark mixed when `torch.version.cuda` is set, `torch.version.hip` is unset, or the local version tag disagrees with the intended ROCm wheel target.

**Phase Placement:** Phase 3, Phase 4.

### 5. Misusing uv Indexes And Lockfiles For A Matrix

**What goes wrong:** The project adds multiple PyTorch ROCm indexes, but uv resolves packages from the wrong index or keeps using one frozen `uv.lock` for all matrix rows.

**Why it happens:** uv indexes are prioritized, `explicit = true` only applies to packages that select the index in `[tool.uv.sources]`, and `uv sync --frozen` will not update the lock. The current config pins `torch` and `torchvision` to `pytorch-rocm71`; that is correct for one row, not a version matrix.

**Consequences:** Builds are reproducible but wrong, or flexible but not reproducible. Dependency confusion risk increases if PyTorch indexes are not explicit and package sources are not pinned.

**Prevention:**

- Decide explicitly between per-row lockfiles, generated constraints, or a matrix-specific lock workflow. Do not pretend one frozen lock can validate multiple PyTorch ROCm wheel targets unless the target wheel is identical by design.
- Keep PyTorch ROCm indexes `explicit = true` and pin `torch`, `torchvision`, and any ROCm-specific companion packages through `[tool.uv.sources]`.
- Record the lockfile path/checksum used for each matrix row.

**Required Guardrails:**

- A lock validation command that verifies pyproject target, uv source index, lockfile package URL/tag, and runtime `torch.__version__` all agree.
- Tests that simulate unavailable indexes/wheels and produce `pytorch_wheel_unavailable`, not an opaque Docker build failure.
- Avoid `--index-strategy unsafe-*` for routine builds unless the report records that exception.

**Phase Placement:** Phase 3, Phase 5.

### 6. Treating Triton ROCm As Version-Neutral

**What goes wrong:** The matrix changes ROCm/PyTorch versions but leaves `triton-rocm==3.6.0` untouched and reports Triton failures as benchmark failures.

**Why it happens:** Triton ROCm is installed from the PyTorch wheel root index today. Its compatibility with a selected PyTorch ROCm wheel and container ROCm user space must be proven, not assumed.

**Consequences:** Triton example failures may be caused by package coordination, not SOL ExecBench behavior.

**Prevention:**

- Include `triton-rocm` version and import/backend readiness in the matrix row.
- Classify Triton-specific failures separately from PyTorch-only and HIP/C++ rows.
- Keep Triton checks in Docker dependency tests and require them before claiming Triton matrix support.

**Required Guardrails:**

- `runtime.evidence` or matrix report must record Triton import status and package version when installed.
- Matrix docs must distinguish "PyTorch ROCm available" from "Triton ROCm available".
- If a ROCm row has no compatible Triton package, mark the Triton path unavailable while preserving PyTorch/HIP row results.

**Phase Placement:** Phase 3, Phase 4.

### 7. Collapsing Host Tools And Container Tools Into One Environment Snapshot

**What goes wrong:** Evidence records `rocminfo`, `amd-smi`, `hipcc`, and PyTorch metadata without saying whether each came from the host or from inside the container.

**Why it happens:** The current environment snapshot was designed for one execution environment. Docker matrix validation has at least two relevant environments: host kernel/driver/device state and container user-space/toolchain state.

**Consequences:** Reports cannot explain mixed-version failures. A user cannot tell whether `hipcc --version` is from `/opt/rocm` in the image or from the host.

**Prevention:**

- Extend evidence with explicit scopes: `host`, `container`, and `python_runtime`.
- Collect host evidence before `docker run` and container evidence inside the image.
- Record command paths, return codes, stdout/stderr tails, and parsed versions for each scope.

**Required Guardrails:**

- Matrix reports must not accept unscoped ROCm version fields.
- The sidecar must include `container_image`, `container_id` or image digest when available, and mounted device list.
- Existing `SOLEXECBENCH_ENV_SNAPSHOT` semantics should stay backward-compatible; matrix evidence can be a new sidecar/report.

**Phase Placement:** Phase 1, Phase 4.

### 8. Letting Docker Runtime Problems Masquerade As Benchmark Regressions

**What goes wrong:** Missing `/dev/kfd`, missing `/dev/dri`, wrong Docker context, missing `video`/`render` permissions, rootless mode, or Docker Desktop causes benchmark/test failures that look like SOL ExecBench regressions.

**Why it happens:** ROCm container execution depends on host device nodes and native Linux Docker behavior. The current wrapper already rejects Docker Desktop contexts and checks device nodes; v1.18 must preserve and expand those checks without hiding real benchmark failures.

**Consequences:** Users waste time debugging benchmark code when the container cannot access the GPU.

**Prevention:**

- Keep wrapper preflight failures before Docker build/run.
- Add a container preflight command that can be run without a benchmark problem directory.
- Separate wrapper errors, container runtime readiness, dependency readiness, benchmark correctness, and performance/timing results in reports.

**Required Guardrails:**

- Matrix rows must have a `preflight` section and benchmark results should be absent when preflight failed.
- Docs must state Docker Desktop/rootless/native-daemon expectations.
- Tests should cover wrapper argument construction and error messaging without needing real Docker.

**Phase Placement:** Phase 2, Phase 4, Phase 5.

### 9. Making CI Promise More Than It Can Run

**What goes wrong:** GitHub Actions or CPU-safe CI is expected to build/run ROCm containers and validate GPU rows.

**Why it happens:** The project already has CPU-safe CI and Docker dependency checks, but ROCm container matrix validation requires AMD GPU device passthrough and a compatible host driver.

**Consequences:** CI either fails consistently, skips the meaningful checks, or creates false confidence through build-only validation.

**Prevention:**

- Keep CPU CI limited to schema, parser, command construction, docs, and lockfile/index policy tests.
- Put live Docker matrix checks behind local ROCm-capable commands or a self-hosted AMD runner.
- Label build-only evidence as `image_build_validated` or `dependency_resolution_validated`, not runtime validated.

**Required Guardrails:**

- A report state for `not_tested` and optionally `build_only`.
- Docs must say remote CI does not prove ROCm runtime unless it runs on AMD GPU hardware with `/dev/kfd` and `/dev/dri`.
- Test names should encode whether they are CPU-safe, Docker-build-only, or ROCm-runtime checks.

**Phase Placement:** Phase 5.

### 10. Reporting One Pass/Fail Instead Of Per-Row Denominator Accounting

**What goes wrong:** The matrix emits a single "passed" or "failed" value for ROCm versions, losing which rows were unavailable, mixed, build-only, not tested, or benchmark-failed.

**Why it happens:** Existing benchmark commands return ordinary test/process status, but a version matrix is a compatibility report with multiple possible non-benchmark outcomes.

**Consequences:** Roadmap and release notes cannot safely claim coverage. Failures become ambiguous.

**Prevention:**

- Use deterministic row identities: host driver/ROCm, container ROCm image, Python version, PyTorch target, Triton target, GPU arch.
- Require row-level states and aggregate counts by state.
- Preserve benchmark command outputs only under rows that passed preflight and dependency checks.

**Required Guardrails:**

- Matrix report schema with `rows[]`, `summary.counts_by_state`, `claim_boundary`, and `evidence_refs`.
- Aggregate status must degrade conservatively: any `mixed_version` or `runtime_unavailable` prevents broad compatibility claims.
- Docs must show examples of partial matrices and how to read them.

**Phase Placement:** Phase 1, Phase 4, Phase 5.

### 11. Ignoring Native HIP/C++ Build Coupling To The Container Toolchain

**What goes wrong:** PyTorch reports a usable ROCm backend, but HIP/C++ extension builds fail because `hipcc`, headers, libraries, or `PYTORCH_ROCM_ARCH` do not match the selected row and GPU.

**Why it happens:** This project builds native extensions through PyTorch. The Docker image supplies `/opt/rocm`, headers, libraries, and compiler tools; PyTorch supplies extension integration and stream APIs. These are related but not identical readiness surfaces.

**Consequences:** HIP/C++ rows fail after PyTorch smoke tests succeed, and the failure is misclassified as a benchmark/kernel issue.

**Prevention:**

- Keep separate readiness checks for PyTorch runtime, HIP compiler, ROCm headers, ROCm libraries, Triton, and benchmark execution.
- Record `hipcc --version`, `/opt/rocm/.info/version*` if present, header availability, `PYTORCH_ROCM_ARCH`, and local gfx target.
- Classify native extension build failures separately from runtime unavailability.

**Required Guardrails:**

- Matrix rows should include `native_build_ready: true/false` with failure reasons.
- Tests should cover generated build command/environment for selected gfx targets.
- Docs should tell users that PyTorch runtime readiness is necessary but not sufficient for HIP/C++ benchmark coverage.

**Phase Placement:** Phase 4, Phase 5.

### 12. Forgetting Claim Guardrails In Release Notes And Docs

**What goes wrong:** Documentation says "ROCm 7.0/7.1/7.2 supported" after only container user-space checks over a ROCm 7.1.x host.

**Why it happens:** Matrix tables are easy to summarize too aggressively. The existing project has strong claim discipline; v1.18 needs the same language around container scope.

**Consequences:** Users infer native host validation, CDNA validation, or full benchmark paper parity that has not been produced.

**Prevention:**

- Add a `docs/CLAIMS.md` row for Docker ROCm user-space matrix evidence.
- Use allowed phrases: "Docker user-space ROCm row validated", "host ROCm/driver X with container ROCm Y", "not native host validation".
- Keep CDNA 3/CDNA 4 and full paper parity guardrails unchanged unless direct evidence is archived.

**Required Guardrails:**

- Documentation no-claim tests for "native ROCm 7.0 validated", "host validated by container", and similar phrases.
- Matrix reports must include a `claim_boundary` object.
- Release closure must list exact commands, host evidence, image tags/digests, row states, and known gaps.

**Phase Placement:** Phase 1, Phase 5.

## Required Guardrails Summary

| Guardrail | Required By | Phase |
| --- | --- | --- |
| Explicit matrix states: `host_validated`, `container_validated`, `mixed_version`, `pytorch_wheel_unavailable`, `runtime_unavailable`, `not_tested` | Prevent overclaiming and ambiguous failures. | Phase 1 |
| Host/container scoped evidence | Explain kernel-driver versus user-space boundary. | Phase 1, Phase 4 |
| Parameterized Docker base image and recorded build args/image tag | Keep rows reproducible. | Phase 2 |
| Docker preflight before benchmark execution | Separate runtime setup failures from benchmark failures. | Phase 2, Phase 4 |
| PyTorch ROCm wheel target validation | Detect CPU/CUDA/mismatched ROCm wheels. | Phase 3, Phase 4 |
| uv explicit indexes and per-row lock strategy | Prevent wrong package sources and frozen-lock drift. | Phase 3 |
| Triton ROCm package/runtime evidence | Avoid treating Triton as automatically covered by PyTorch. | Phase 3, Phase 4 |
| Row-level compatibility report with denominator accounting | Make partial coverage auditable. | Phase 4 |
| CPU-safe tests plus optional ROCm Docker checks | Keep CI honest without requiring unavailable GPUs. | Phase 5 |
| Documentation no-claim tests | Prevent Docker user-space evidence from becoming native host claims. | Phase 5 |

## Recommended Roadmap Shape

1. **Phase 1: Matrix Contract And Status Vocabulary**
   - Define the sidecar/report schema, row states, evidence scopes, and claim boundary first. This prevents later phases from producing evidence that cannot be safely interpreted.

2. **Phase 2: Docker Image Selection And Build Args**
   - Parameterize the Dockerfile and wrapper after the contract is known. Include preflight and image metadata early because every later phase depends on reproducible row identity.

3. **Phase 3: uv And PyTorch ROCm Wheel Coordination**
   - Solve dependency selection before runtime validation. A matrix is meaningless if `uv sync --frozen` silently installs the wrong wheel for a row.

4. **Phase 4: Runtime Evidence And Compatibility Reports**
   - Extend evidence collection and generate row-level reports. This is where host/container/PyTorch/Triton/toolchain fields are cross-checked and row states are assigned.

5. **Phase 5: Validation Workflow, Docs, And CI Guardrails**
   - Add CPU-safe tests, Docker runtime instructions, no-claim docs tests, and release-closure evidence. Keep live GPU/container checks out of ordinary CI unless a self-hosted AMD runner exists.

## Sources

- AMD Container Runtime Toolkit requirements: https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/requirements.html
- AMD amdgpu compatibility matrix: https://instinct.docs.amd.com/projects/amdgpu-docs/en/docs-30.20.1/compatibility/compatibility-matrix.html
- AMD ROCm Docker installation docs: https://rocm.docs.amd.com/_/downloads/install-on-linux/en/latest/pdf/
- PyTorch previous versions: https://pytorch.org/get-started/previous-versions
- uv package indexes: https://docs.astral.sh/uv/concepts/indexes/
- uv dependency sources: https://docs.astral.sh/uv/concepts/projects/dependencies/
