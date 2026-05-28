# Project Research Summary

**Project:** SOL ExecBench ROCm Port
**Domain:** Docker-based ROCm user-space version compatibility matrix
**Milestone:** v1.18 ROCm Version Matrix via Docker
**Researched:** 2026-05-28
**Confidence:** MEDIUM-HIGH

## Executive Summary

v1.18 should add a Docker-based ROCm compatibility matrix for SOL ExecBench without changing benchmark semantics. This is diagnostic infrastructure: users select a ROCm container user-space row, coordinate the intended PyTorch ROCm wheel policy, run bounded runtime/toolchain probes, and archive machine-readable evidence that says exactly what was validated.

The recommended approach is sidecar-first and conservative. Keep the current ROCm 7.1 Docker and `uv.lock` path as the default baseline, add an explicit matrix manifest for ROCm 7.0/7.1/7.2 rows, parameterize Docker build/run selection, and introduce a `sol_execbench.rocm_compatibility_matrix.v1` report separate from canonical trace JSONL. Docker scripts should provide requested coordinates; Python probes should record observed container, host, PyTorch, Triton, toolchain, and GPU evidence; the compatibility model should classify the row.

The central risk is overclaiming. A Docker pass validates container ROCm user-space on the current host driver/devices; it does not prove native host ROCm validation, paper parity, score authority, or leaderboard eligibility. Mitigate this with strict status vocabulary, scoped evidence fields, claim-boundary booleans, stable reason codes, docs wording checks, and no silent fallback from missing PyTorch ROCm wheels to CPU/CUDA/wrong-index packages.

## Key Findings

### Stack Additions

v1.18 does not need a new package manager, service, or evaluator backend. It should extend the existing Python 3.12, Pydantic v2, Click/Rich, Docker, uv, PyTorch ROCm, Triton ROCm, and ROCm toolchain stack.

**Core technologies:**
- Docker ROCm selector: parameterize `docker/Dockerfile` and `scripts/run_docker.sh` with `ROCM_VERSION`, exact image/tag, and build/run marker env vars.
- Matrix manifest: add a small checked-in source of truth, likely `docker/rocm-version-matrix.toml`, for logical rows, Docker tags, PyTorch ROCm index policy, expected wheel suffix, and default evidence state.
- uv explicit indexes: keep PyTorch-family packages pinned to explicit ROCm indexes and keep PyPI as the generic default; do not rely on index order.
- Compatibility models: add strict Pydantic report models, likely in a new `src/sol_execbench/core/compatibility.py`, while preserving `EnvironmentSnapshot` semantics.
- Runtime probes: reuse/extend `doctor --json` and environment probe patterns for `hipcc`, `rocminfo`, `rocm_agent_enumerator`, optional `rocprofv3`, `torch.__version__`, `torch.version.hip`, `torch.version.cuda`, device availability, device name, and `gcnArchName`.
- Reports: emit standalone or sidecar JSON plus concise human-readable output; do not mutate trace JSONL.

Critical version strategy: keep ROCm 7.1 as the canonical default lock until a per-row lock workflow is proven. Initial Docker rows should be exact `rocm/dev-ubuntu-24.04:*complete` tags for 7.0.x, 7.1.x, and 7.2.x, with 7.1 remaining the default.

### Expected Features

**Must have (table stakes):**
- Select a configured ROCm Docker user-space version while preserving today’s default `./scripts/run_docker.sh --build` behavior.
- Record requested vs observed container ROCm, host ROCm/driver context, PyTorch ROCm wheel identity, Triton ROCm state, toolchain evidence, GPU architecture, uv index/lock policy, and artifact refs.
- Emit `compatibility-matrix.json` with one row per configured target and aggregate counts by state.
- Emit per-entry `compatibility.json` sidecars with probe data, command/log refs, status, reason codes, and claim flags.
- Support exact statuses: `host_validated`, `container_validated`, `mixed_version`, `pytorch_wheel_unavailable`, `runtime_unavailable`, `not_tested`.
- Distinguish missing PyTorch wheels from runtime/device failures.
- Detect CPU/CUDA/wrong-ROCm PyTorch wheels and classify them conservatively.
- Preserve canonical trace JSONL, correctness, timing, scoring, paper-parity, and leaderboard semantics.
- Provide CPU-safe unit/fixture tests for classification, serialization, command construction, and report rendering.
- Update docs with explicit container-vs-host claim boundaries.

**Should have (differentiators):**
- Docker image digest capture for reproducible evidence.
- Native-host comparison row that can produce `host_validated` only from direct host probes.
- Wheel availability preflight before expensive Docker builds.
- JSON schema export or CLI schema output for external consumers.
- Matrix diff/report helpers and per-architecture aggregation when evidence exists.
- Bounded command transcript capture for Docker, uv, and probe commands.

**Defer (v2+):**
- Native ROCm host install/reinstall automation.
- Full paper dataset validation across every ROCm version.
- Score, leaderboard, or paper-parity policy changes.
- CUDA/NVIDIA matrix support.
- Arbitrary ROCm versions outside the declared manifest.
- Docker-in-Docker, conda/mamba, Spack, Nix, or host driver management.
- Full CDNA 3/CDNA 4 claims without archived hardware evidence.

### Architecture Approach

Keep Docker orchestration at the script/Dockerfile boundary and compatibility classification inside Python. The evaluator should not call Docker, and Docker scripts should not be the source of truth for validation. Instead, scripts pass requested matrix coordinates into the container; Python probes collect observed evidence; compatibility models compare requested and observed values and write diagnostic reports.

**Major components:**
1. Docker matrix configuration: checked-in rows for Docker image/tag, PyTorch ROCm index/tag policy, support status, and default claim state.
2. Docker wrapper plumbing: `scripts/run_docker.sh --rocm` or equivalent, build args, image tags, native Linux Docker/device preflight, and container marker env vars.
3. uv/PyTorch wheel coordination: explicit package-to-index policy, default ROCm 7.1 lock, per-row availability checks, and unavailable classification.
4. Compatibility evidence models: scoped host/container/python/toolchain/GPU evidence, row status, reason codes, authority flags, and matrix summary counts.
5. CLI/report surface: `sol-execbench compatibility --json` or `doctor --json` extension, optional trace-adjacent sidecar, and Markdown/Rich summary.
6. Docs/tests guardrails: docs claim language, CPU-safe fixture tests, Docker script tests, and marker-gated live ROCm checks.

### Critical Pitfalls

1. **Treating container ROCm as host validation** - use `validation_scope`, scoped host/container evidence, and wording such as "container ROCm user-space validated on host driver X"; reserve `host_validated` for direct native runs.
2. **Assuming any ROCm container works with the current host driver** - preflight Docker context, `/dev/kfd`, `/dev/dri`, `rocminfo`, PyTorch device visibility, memory copy, and event timing before benchmark execution.
3. **Hardcoding image tags in multiple places** - use one matrix manifest plus Docker build args/env labels, and store exact image tag/digest/build args in reports.
4. **Mixing PyTorch ROCm wheels and container user-space silently** - compare intended wheel target with `torch.__version__`, `torch.version.hip`, `torch.version.cuda`, and container ROCm evidence; classify mismatches as `mixed_version`.
5. **Misusing uv indexes/lockfiles** - keep indexes explicit, do not pretend one frozen lock validates multiple wheel targets, and record per-row lock/index evidence.
6. **Letting runtime setup failures look like benchmark regressions** - classify missing devices, Docker Desktop/rootless issues, unavailable wheels, and failed preflights before running benchmarks.
7. **Reporting one pass/fail for the matrix** - require row-level denominators and aggregate counts; partial matrices must be readable and conservative.

## Implications For Roadmap

Based on research, suggested phase structure:

### Phase 1: Matrix Contract And Claim Guardrails

**Rationale:** Every later phase depends on stable report semantics and claim boundaries.
**Delivers:** Pydantic status enums, reason-code taxonomy, `sol_execbench.rocm_compatibility_matrix.v1`, claim-boundary booleans, row/matrix summary models, fixture tests, and docs wording requirements.
**Addresses:** Compatibility states, diagnostic-only authority, `not_tested` rows, aggregate counts, and container-vs-host wording.
**Avoids:** Overclaiming Docker runs, trace/schema mutation, boolean-only pass/fail reports, and unscoped evidence.

### Phase 2: Docker Matrix Selection And Preflight

**Rationale:** Users need deterministic image selection before wheel/runtime evidence can be meaningful.
**Delivers:** Matrix manifest, parameterized Dockerfile base image, `scripts/run_docker.sh` ROCm selector, build/run marker env vars, exact image tag recording, existing default preservation, Docker context/device checks, and unknown-row rejection.
**Uses:** Docker `ARG`, `rocm/dev-ubuntu-24.04:*complete` images, `/dev/kfd`, `/dev/dri`, native Linux Docker checks.
**Avoids:** Drift between docs/scripts/Dockerfile, Docker Desktop/rootless false failures, floating-tag claims, and unsupported arbitrary rows.

### Phase 3: uv And PyTorch ROCm Wheel Coordination

**Rationale:** The matrix is invalid if the selected container installs the wrong PyTorch backend.
**Delivers:** Explicit PyTorch ROCm index rows, default ROCm 7.1 lock policy, wheel availability/preflight helper, lock/index evidence capture, local-version parsing, and `pytorch_wheel_unavailable` / `mixed_version` classification tests.
**Uses:** uv explicit indexes and `[tool.uv.sources]`, PyTorch ROCm wheel indexes, `torch.__version__`, `torch.version.hip`, `torch.version.cuda`.
**Avoids:** CPU/CUDA fallback, wrong-index resolution, assuming one `uv.lock` covers all rows, and hiding unavailable wheels as Docker build failures.

### Phase 4: Runtime Evidence And Compatibility Reports

**Rationale:** Once requested coordinates and dependency policy exist, runtime probes can produce useful compatibility evidence.
**Delivers:** Scoped host/container/python/toolchain/GPU probe collection, `compatibility --json` or `doctor --json` report surface, per-entry sidecars, matrix aggregate JSON, human report, optional smoke hook, and report artifact refs.
**Implements:** Compatibility model comparison of requested-vs-observed container ROCm, host context, PyTorch ROCm tag, Triton readiness, native HIP/C++ build readiness, and GPU architecture.
**Avoids:** Collapsing host/container tools, treating Triton as version-neutral, PyTorch-only smoke overclaims, and benchmark regressions caused by preflight failures.

### Phase 5: Validation Workflow, Docs, And CI Guardrails

**Rationale:** The matrix is only useful if users and release notes can read it without overstating coverage.
**Delivers:** Documentation in `docs/CONFIGURATION.md`, `docs/TESTING.md`, `docs/rocm.md`, `docs/CLAIMS.md`, likely `docs/rocm_version_matrix.md`; CPU-safe tests; Dockerfile/script tests; marker-gated live ROCm tests for the default row; optional self-hosted AMD runner guidance; archived validation examples.
**Addresses:** Claim language, CI limits, partial matrix interpretation, hardware-specific evidence boundaries, and remediation hints.
**Avoids:** CI promising GPU validation it cannot run, build-only results being called runtime validation, and one architecture being generalized to all AMD GPUs.

### Phase Ordering Rationale

- Contract and claim boundaries come first because they define what every report can safely mean.
- Docker selection must precede runtime evidence because evidence rows need deterministic requested coordinates.
- uv/PyTorch coordination must land before live runtime claims because a ROCm container with the wrong wheel is a mixed stack, not a validated row.
- Runtime report generation comes after probes and selectors so it can compare requested vs observed data.
- Docs and validation close the loop because the main product value is trustworthy, auditable claim wording.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** exact PyTorch ROCm wheel availability for ROCm 7.0 and 7.2, Python 3.12/3.13 compatibility, and whether per-row locks or generated constraints are least costly.
- **Phase 4:** live Triton ROCm compatibility, native HIP/C++ extension build readiness, and host-driver/container compatibility behavior across selected image tags.
- **Phase 5:** CI/self-hosted AMD runner feasibility and any CDNA 3 validation plan.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Pydantic sidecar contract, status enums, authority flags, and trace isolation follow existing repository patterns.
- **Phase 2:** Dockerfile/script parameterization and device preflight are well-defined from current wrapper behavior.

## Explicit Claim Boundary Language

Use this wording in docs, reports, and release notes:

- Passing Docker rows are **container ROCm user-space validated** on the recorded host driver/devices.
- A Docker row is **not native host ROCm validated** unless the report came from a direct host validation mode.
- Compatibility sidecars are **diagnostic compatibility evidence**. They are not correctness, timing, score, paper-parity, or leaderboard authority.
- `torch.cuda` API names may appear in ROCm evidence because PyTorch exposes AMD GPUs through the CUDA-compatible namespace; this must not be described as CUDA/NVIDIA runtime support.
- `mixed_version` means the requested ROCm row, container user-space, PyTorch ROCm wheel, Triton ROCm package, or host evidence do not align enough for a clean compatibility claim.
- `pytorch_wheel_unavailable` means the dependency stack could not be resolved for the requested row; it is not a benchmark failure.
- `runtime_unavailable` means the container/dependency stack could not access the required ROCm runtime or GPU devices; benchmark results should be absent or clearly separated.
- Hardware validation is architecture-specific and requires archived device evidence for that `gfx*` target.

Required claim flags for Docker compatibility entries:

| Flag | Required Behavior |
|------|-------------------|
| `container_user_space_validated` | `true` only when the selected Docker entry passed required probes. |
| `native_host_validated` | `true` only for direct host validation; normally `false` for Docker rows. |
| `hardware_validated` | `true` only when real device evidence for that architecture is archived. |
| `paper_parity_authority` | Always `false` for v1.18 compatibility sidecars. |
| `score_authority` | Always `false`. |
| `leaderboard_authority` | Always `false`. |
| `diagnostic_compatibility_evidence` | Always `true` for these sidecars. |

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Repo integration points are clear; exact Docker tags and PyTorch ROCm wheel availability must be verified during implementation. |
| Features | HIGH | Table stakes, statuses, sidecars, and claim boundaries are strongly supported by repo patterns and milestone scope. |
| Architecture | HIGH | Existing CLI, environment sidecars, doctor diagnostics, Docker wrapper, and reporting boundaries give a clear integration path. |
| Pitfalls | HIGH | Risks are grounded in current code, Docker/ROCm runtime behavior, uv semantics, and existing claim-guardrail docs. |

**Overall confidence:** MEDIUM-HIGH

### Gaps To Address

- PyTorch ROCm wheel matrix: verify available torch/torchvision/triton-rocm versions for ROCm 7.0, 7.1, and 7.2 with Python `>=3.12,<3.14`.
- Lockfile workflow: decide whether v1.18 uses only default ROCm 7.1 lock plus per-row checks, generated lock artifacts, or constraints.
- Host-driver compatibility: record actual host driver/runtime evidence and validate which selected containers can run on the available host.
- Triton ROCm readiness: test separately from PyTorch import/device availability.
- Image digest capture: determine how reliably `scripts/run_docker.sh` can capture resolved digests for built/pulled images.
- CDNA 3 coverage: keep claims deferred unless live CDNA 3 evidence is archived.

## Sources

### Primary (HIGH confidence)

- `.planning/research/STACK.md` - stack additions, version/dependency strategy, integration boundaries, risks, and exclusions.
- `.planning/research/FEATURES.md` - table stakes, differentiators, out-of-scope items, status vocabulary, reason codes, and acceptance signals.
- `.planning/research/ARCHITECTURE.md` - current integration points, proposed components, data flow, build order, and test strategy.
- `.planning/research/PITFALLS.md` - critical pitfalls, phase placement, guardrails, and verification warnings.
- Local repo files referenced by research: `pyproject.toml`, `docker/Dockerfile`, `scripts/run_docker.sh`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/cli/main.py`, `docs/CLAIMS.md`, `docs/CONFIGURATION.md`, `docs/TESTING.md`, `docs/rocm.md`, `tests/docker/dependencies/`.

### External (MEDIUM/HIGH confidence)

- AMD ROCm Docker and container runtime documentation - host kernel/driver sharing, `/dev/kfd`, `/dev/dri`, Docker Desktop/rootless limitations, and driver/runtime compatibility caveats.
- AMD Docker Hub `rocm/dev-ubuntu-24.04` tags - available ROCm dev image tags for selected rows.
- PyTorch previous versions and ROCm wheel-index docs - ROCm wheel index patterns and current/default ROCm 7.1 install commands.
- uv documentation - explicit indexes, package-to-index pinning, and index strategy behavior.
- AMD ROCm PyTorch install docs - PyTorch/ROCm Docker install examples and validated image inventories.

---
*Research completed: 2026-05-28*
*Ready for roadmap: yes*
