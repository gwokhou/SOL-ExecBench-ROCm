# Phase 79: Docker Matrix Selection And Preflight - Research

**Researched:** 2026-05-28  
**Domain:** ROCm Docker target selection, Dockerfile parameterization, runtime preflight classification  
**Confidence:** HIGH for repository architecture and Docker/ROCm runtime boundaries; MEDIUM for live Docker Hub tag availability because tags were checked by web crawl but not pulled locally.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Implementation Decisions

### Docker Target Manifest Scope
- Use a checked-in manifest of declared Docker Targets rather than ad hoc
  discovery of arbitrary ROCm image tags.
- Include logical ROCm Target entries for configured 7.0.x, 7.1.x, and 7.2.x
  where usable Docker image tags are known.
- Preserve the current ROCm 7.1 Docker behavior as the default path so existing
  `./scripts/run_docker.sh` and `docker/Dockerfile` usage continues to work.
- Reject unknown Targets unless the user supplies an explicit unsafe or
  untested override.

### Docker Preflight Classification
- Classify Docker Desktop or otherwise unsupported Docker contexts as
  `runtime_unavailable` before benchmark execution.
- Classify missing `/dev/kfd`, missing `/dev/dri`, and inaccessible GPU device
  access as `runtime_unavailable` before benchmark execution.
- Preflight failures should produce diagnostic Matrix-compatible evidence and
  should not continue into benchmark execution.
- These classifications are compatibility evidence only and must not imply
  correctness, timing, scoring, paper-parity, or leaderboard authority.

### Docker Evidence Fields
- Record exact requested image repository and image tag for the selected Target.
- Record resolved image digest when available, but treat digest resolution as
  best-effort and non-blocking.
- Record Docker build arguments used for the selected Target, including ROCm
  base-image selection inputs.
- Record selected Target id, validation scope, and runtime preflight details so
  downstream phases can emit full Matrix Entries and aggregate reports.

### Unsafe Or Untested Override Boundaries
- Unknown Target overrides may allow diagnostic probes or smoke-style setup
  checks only.
- Unknown or untested override paths must not emit `container_validated`,
  `host_validated`, benchmark eligibility, score authority, paper-parity
  authority, or leaderboard authority.
- Use Phase 78 `mixed_version`, `runtime_unavailable`, and `not_tested`
  semantics where applicable instead of inventing new status words.
- Keep override naming explicit enough that logs, tests, and docs cannot present
  the path as normal validation.

### the agent's Discretion
The agent may choose the exact manifest format, helper module names, and script
flag spelling as long as the result is checked in, auditable, CPU-testable, and
keeps the existing ROCm 7.1 default behavior stable.

### Deferred Ideas (OUT OF SCOPE)
## Deferred Ideas

- PyTorch ROCm wheel local-version tags, uv index selection, and mixed-version
  dependency policy belong to Phase 80.
- Full host/container/Python/toolchain/GPU runtime evidence collection and
  aggregate compatibility report emission belong to Phase 81.
- User-facing validation workflow docs, CI guardrails, and live marker-gated
  ROCm validation guidance belong to Phase 82.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DOCKER-01 | Checked-in ROCm Docker matrix manifest for declared configured 7.0.x, 7.1.x, and 7.2.x logical Targets where usable Docker image tags are known. [VERIFIED: .planning/REQUIREMENTS.md] | Use a repo-owned JSON or TOML manifest converted to Phase 78 `MatrixTarget` objects; do not discover arbitrary tags at runtime. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md] |
| DOCKER-02 | `docker/Dockerfile` supports parameterized ROCm base image selection while preserving the current ROCm 7.1 default behavior. [VERIFIED: .planning/REQUIREMENTS.md] | Docker supports `ARG` before `FROM`, and the current default base is `rocm/dev-ubuntu-24.04:7.1.1-complete`. [CITED: https://docs.docker.com/reference/dockerfile] [VERIFIED: docker/Dockerfile] |
| DOCKER-03 | `scripts/run_docker.sh` supports selecting a declared ROCm Target and rejects unknown Targets unless explicit unsafe/untested override is supplied. [VERIFIED: .planning/REQUIREMENTS.md] | Keep `scripts/run_docker.sh` as the entry point, but delegate selection/classification to Python helpers so unknown-target behavior is unit-testable without Docker. [VERIFIED: scripts/run_docker.sh] |
| DOCKER-04 | Docker preflight checks classify missing `/dev/kfd`, missing `/dev/dri`, unsupported Docker context, and inaccessible GPU devices as `runtime_unavailable` before benchmark execution. [VERIFIED: .planning/REQUIREMENTS.md] | Reuse Phase 78 `MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE` and `MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE`; AMD docs require `/dev/kfd` and `/dev/dri` for ROCm containers. [VERIFIED: src/sol_execbench/core/compatibility.py] [CITED: https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.0.0/how-to/docker.html] |
| DOCKER-05 | Docker reports record requested image repository/tag and, when available, resolved image digest and build arguments. [VERIFIED: .planning/REQUIREMENTS.md] | `MatrixTarget` already stores requested repository/tag; `MatrixContainerEvidence` stores digest; add a Phase 79 build-arg/evidence helper or artifact payload for build args. [VERIFIED: src/sol_execbench/core/compatibility.py] |
</phase_requirements>

## Summary

Phase 79 should stay local and deterministic: add a checked-in Docker Target manifest, add a pure Python selector/preflight module that maps manifest entries into Phase 78 `MatrixTarget` and non-authoritative Matrix evidence, then keep `scripts/run_docker.sh` as a thin wrapper around that selector. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md] [VERIFIED: scripts/run_docker.sh] The existing wrapper already owns Docker context checks, `/dev/kfd`, `/dev/dri`, build/run flags, and default shell behavior, but it exits with shell errors instead of returning Matrix-compatible classifications. [VERIFIED: scripts/run_docker.sh]

The Dockerfile should use pre-`FROM` build args for `ROCM_DOCKER_IMAGE` and `ROCM_DOCKER_TAG`, with defaults matching `rocm/dev-ubuntu-24.04:7.1.1-complete`. [VERIFIED: docker/Dockerfile] [CITED: https://docs.docker.com/reference/dockerfile] Docker Hub currently lists `rocm/dev-ubuntu-24.04` tags for `7.0.2-complete`, `7.1.1-complete`, and `7.2.x-complete`, but Phase 79 should record these as declared/auditable requested tags, not as live compatibility claims. [CITED: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags] [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md]

**Primary recommendation:** implement `src/sol_execbench/core/docker_matrix.py` plus `docker/rocm-targets.json`; parameterize `docker/Dockerfile` with default-preserving `ARG`s; update `scripts/run_docker.sh` with `--target`, `--allow-unknown-target`, and `--preflight-only`; cover all behavior with CPU-safe tests under `tests/sol_execbench/`. [VERIFIED: AGENTS.md] [ASSUMED]

## Project Constraints (from AGENTS.md)

- Python package code belongs under `src/sol_execbench/`; CLI entry point is `sol_execbench.cli:cli`; tests belong under `tests/`, with package tests under `tests/sol_execbench/`. [VERIFIED: AGENTS.md]
- Use `uv sync --all-groups`, `uv run pytest tests/`, targeted pytest files, Ruff, and `./scripts/run_docker.sh --build` as project commands. [VERIFIED: AGENTS.md]
- Use Python 3.12+, Ruff style, `snake_case` functions/modules, `PascalCase` classes/Pydantic models, and descriptive `test_*` names. [VERIFIED: AGENTS.md]
- New tests should sit near related coverage under `tests/sol_execbench/`; use existing ROCm markers for GPU-sensitive behavior. [VERIFIED: AGENTS.md]
- Do not commit proprietary kernels, credentials, Hugging Face tokens, downloaded datasets, local cache, build output, or benchmark output. [VERIFIED: AGENTS.md]
- Docker/GPU assumptions may require Docker, ROCm-capable AMD hardware, ROCm drivers, and access to `/dev/kfd` and `/dev/dri`; document hardware-specific assumptions in tests or PR notes. [VERIFIED: AGENTS.md]
- GSD workflow enforcement says repo edits should go through GSD entry points unless explicitly bypassed. [VERIFIED: AGENTS.md]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Declared Docker Target manifest | Repository config / Core Python | Docker wrapper | The manifest is source-controlled policy, while Python can parse and validate it into Phase 78 `MatrixTarget` objects. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md] |
| Target selection and unknown-target rejection | Core Python | Shell wrapper | Pure Python selection makes default preservation, aliases, override classification, and evidence construction CPU-testable. [VERIFIED: AGENTS.md] [ASSUMED] |
| Docker base image selection | Dockerfile | Shell wrapper | Dockerfile owns the `FROM` image; wrapper passes build args selected from the manifest. [VERIFIED: docker/Dockerfile] [VERIFIED: scripts/run_docker.sh] |
| Docker context and device preflight | Shell wrapper | Core Python classifier | Shell can inspect host state; Python should classify host observations into bounded Matrix statuses without executing benchmarks. [VERIFIED: scripts/run_docker.sh] [VERIFIED: src/sol_execbench/core/compatibility.py] |
| Compatibility evidence vocabulary | Core Python | Reports in later phases | Phase 78 already owns `MatrixTarget`, `MatrixObservedEvidence`, statuses, reason codes, claim boundaries, and execution decisions. [VERIFIED: src/sol_execbench/core/compatibility.py] |
| Image digest capture | Shell wrapper / Docker CLI | Core Python evidence model | Digest lookup depends on Docker CLI availability and network/local image metadata; evidence should be nullable and non-blocking. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md] |

## Standard Stack

### Core

| Library / Tool | Version | Purpose | Why Standard |
|----------------|---------|---------|--------------|
| Python stdlib `json` or `tomllib` | Python >=3.12 [VERIFIED: pyproject.toml] | Parse the checked-in manifest. [ASSUMED] | Avoids adding dependencies; `tomllib` is available in Python 3.12 and JSON needs no parser dependency. [VERIFIED: pyproject.toml] [ASSUMED] |
| Pydantic v2 project models | `pydantic>=2.12.5` [VERIFIED: pyproject.toml] | Strict immutable manifest/selection/evidence models. [VERIFIED: src/sol_execbench/core/compatibility.py] | Phase 78 uses strict Pydantic v2 models with `extra="forbid"`, `frozen=True`, and enum coercion. [VERIFIED: src/sol_execbench/core/compatibility.py] |
| Docker CLI | local `29.4.3` [VERIFIED: docker --version] | Build/run selected ROCm image and inspect Docker context/digests. [VERIFIED: scripts/run_docker.sh] | Existing workflow uses Docker CLI through `scripts/run_docker.sh`. [VERIFIED: scripts/run_docker.sh] |
| AMD ROCm dev Docker image | `rocm/dev-ubuntu-24.04:7.1.1-complete` default [VERIFIED: docker/Dockerfile] | ROCm user-space/toolchain base image. [VERIFIED: docker/Dockerfile] | Preserves current 7.1 default while allowing declared 7.0.x and 7.2.x requested tags. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md] |

### Supporting

| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| Pytest | `9.0.2` [VERIFIED: uv run pytest --version] | CPU-safe unit tests for manifest parsing, target selection, preflight classification, and script command construction. [VERIFIED: pyproject.toml] | Per task and phase validation. [VERIFIED: AGENTS.md] |
| Ruff | `0.15.14` [VERIFIED: uv run ruff --version] | Lint Python implementation and tests. [VERIFIED: pyproject.toml] | Before committing implementation. [VERIFIED: AGENTS.md] |
| Docker Hub tag page | updated 2026-05 crawl [CITED: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags] | Audit declared image tags. [CITED: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags] | Research/reference only; implementation should not dynamically depend on the tag listing. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| JSON manifest [ASSUMED] | TOML manifest parsed by `tomllib` [ASSUMED] | TOML is more readable and stdlib in Python 3.12, but JSON is simpler for shell-adjacent tooling and strict fixture comparison. [VERIFIED: pyproject.toml] [ASSUMED] |
| Python-owned selector [ASSUMED] | Bash associative arrays in `scripts/run_docker.sh` [ASSUMED] | Bash is smaller but harder to test deeply; Python aligns with Phase 78 Pydantic contracts. [VERIFIED: src/sol_execbench/core/compatibility.py] [ASSUMED] |
| `--target` flag [ASSUMED] | `ROCM_TARGET` or `ROCM_VERSION` env var [ASSUMED] | A flag is explicit in command history and tests; env vars are easy for CI but easier to miss in logs. [ASSUMED] |

**Installation:** No new external packages are required. [VERIFIED: pyproject.toml]

**Version verification:** Local verification used `uv run pytest --version`, `uv run ruff --version`, `docker --version`, `docker info`, and Docker Hub tag page inspection. [VERIFIED: command output] [CITED: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags]

## Package Legitimacy Audit

No external packages should be installed for this phase. [VERIFIED: pyproject.toml] The planner should avoid package-install tasks unless implementation discovers a hard blocker; if a new package is proposed later, run the package legitimacy protocol before planning install steps. [ASSUMED]

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| None | — | — | — | — | — | No new package install. [VERIFIED: pyproject.toml] |

**Packages removed due to slopcheck [SLOP] verdict:** none. [VERIFIED: pyproject.toml]  
**Packages flagged as suspicious [SUS]:** none. [VERIFIED: pyproject.toml]

## Architecture Patterns

### System Architecture Diagram

```text
User command
  |
  v
scripts/run_docker.sh --target <id> [--allow-unknown-target] [--preflight-only]
  |
  +--> Python selector/classifier
  |      |
  |      +--> docker/rocm-targets.json
  |      |      |
  |      |      +--> Phase 78 MatrixTarget fields
  |      |
  |      +--> unknown target?
  |             |-- no override --> not_tested / reject before build
  |             `-- explicit override --> not_tested, probes/smoke only
  |
  +--> host preflight observations
  |      |
  |      +--> docker context show/inspect
  |      +--> /dev/kfd exists/access
  |      +--> /dev/dri exists/access
  |      `--> optional docker probe for GPU visibility
  |
  +--> classify preflight
  |      |-- unavailable --> runtime_unavailable Matrix-compatible evidence, stop
  |      `-- available --> build/run may proceed
  |
  +--> docker build --build-arg ROCM_DOCKER_IMAGE --build-arg ROCM_DOCKER_TAG
  |
  `--> docker run selected image with ROCm device passthrough
```

### Recommended Project Structure

```text
docker/
├── Dockerfile              # Parameterized ROCm base image, default 7.1.1-complete.
└── rocm-targets.json       # Checked-in declared Docker Matrix Targets.

src/sol_execbench/core/
├── compatibility.py        # Existing Phase 78 Matrix models and execution classifier.
└── docker_matrix.py        # New manifest models, selector, build args, preflight classification.

tests/sol_execbench/
├── test_docker_matrix_targets.py      # Manifest shape, defaults, MatrixTarget conversion.
└── test_docker_matrix_preflight.py    # CPU-safe context/device/unknown-target classification.
```

### Pattern 1: Default-Preserving Dockerfile ARGs

**What:** Put `ARG` declarations before `FROM`, default them to the existing base image, and repeat the args after `FROM` if later Dockerfile instructions need them. [CITED: https://docs.docker.com/reference/dockerfile] [VERIFIED: docker/Dockerfile]

**When to use:** Use for DOCKER-02 so `./scripts/run_docker.sh --build` still builds ROCm 7.1.1 by default while target selection can override `ROCM_DOCKER_TAG`. [VERIFIED: .planning/REQUIREMENTS.md]

**Example:**

```dockerfile
# Source: Dockerfile reference, existing docker/Dockerfile
ARG ROCM_DOCKER_IMAGE=rocm/dev-ubuntu-24.04
ARG ROCM_DOCKER_TAG=7.1.1-complete
FROM ${ROCM_DOCKER_IMAGE}:${ROCM_DOCKER_TAG} AS base
ARG ROCM_DOCKER_IMAGE
ARG ROCM_DOCKER_TAG
```

### Pattern 2: Manifest To Phase 78 Target

**What:** Store requested values in a checked-in manifest and convert each declared entry to the existing strict `MatrixTarget` model. [VERIFIED: src/sol_execbench/core/compatibility.py] [VERIFIED: .planning/REQUIREMENTS.md]

**When to use:** Use for DOCKER-01 and DOCKER-03 so the script selects known Target ids and never fabricates clean validation entries for unknown tags. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md]

**Example:**

```python
# Source: existing MatrixTarget model in src/sol_execbench/core/compatibility.py
MatrixTarget(
    target_id=entry["target_id"],
    requested_rocm_user_space_version=entry["requested_rocm_user_space_version"],
    docker_image_repository=entry["docker_image_repository"],
    docker_image_tag=entry["docker_image_tag"],
    pytorch_rocm_target=entry.get("pytorch_rocm_target"),
    validation_scope=MatrixValidationScope.CONTAINER_USER_SPACE,
    intended_gpu_architecture=entry.get("intended_gpu_architecture"),
)
```

### Pattern 3: Pure Preflight Classifier

**What:** Represent Docker context/device observations as data, then classify them into Phase 78 statuses and reason codes without running Docker or benchmarks. [VERIFIED: src/sol_execbench/core/compatibility.py] [ASSUMED]

**When to use:** Use for DOCKER-04 and CPU-safe tests. [VERIFIED: .planning/REQUIREMENTS.md]

**Example:**

```python
# Source: Phase 78 Matrix status and reason-code vocabulary
if context_is_desktop or missing_kfd or missing_dri or gpu_inaccessible:
    status = MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE
    reason_code = MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE
```

### Anti-Patterns to Avoid

- **Runtime tag discovery as support policy:** Do not turn Docker Hub search results into supported Targets at runtime; the user locked a checked-in manifest. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md]
- **Shell-only compatibility logic:** Do not bury target selection and classification in Bash branches that cannot be unit-tested. [ASSUMED]
- **Overclaiming Docker evidence:** Do not label Docker user-space success as native host validation; Phase 78 rejects container-scoped `host_validated`. [VERIFIED: src/sol_execbench/core/compatibility.py]
- **Digest lookup as a hard dependency:** Do not fail the target solely because digest resolution is unavailable; context explicitly says digest is best-effort. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Matrix status vocabulary | New strings such as `docker_unavailable` or `unknown_target` | Phase 78 `MatrixCompatibilityStatus` and `MatrixCompatibilityReasonCode` | The bounded vocabulary is already locked and tested. [VERIFIED: src/sol_execbench/core/compatibility.py] |
| Matrix target/evidence schema | A separate Docker-only report contract | Existing `MatrixTarget`, `MatrixObservedEvidence`, `MatrixContainerEvidence`, `MatrixClaimBoundary` | Downstream phases depend on the v1 Matrix contract. [VERIFIED: .planning/phases/78-matrix-contract-and-claim-guardrails/78-01-SUMMARY.md] |
| Dockerfile variable interpolation | Custom template generation | Dockerfile pre-`FROM` `ARG` | Official Dockerfile syntax supports args in `FROM`. [CITED: https://docs.docker.com/reference/dockerfile] |
| Docker context detection | Parsing arbitrary Docker config files | `docker context show` and `docker context inspect` | Docker documents active context and inspect behavior. [CITED: https://docs.docker.com/engine/manage-resources/contexts/] |
| ROCm GPU device passthrough | Custom privileged containers or NVIDIA `--gpus` flags | `/dev/kfd`, `/dev/dri`, targeted group/security options | AMD documents `/dev/kfd` and DRI devices for ROCm containers. [CITED: https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.0.0/how-to/docker.html] |

**Key insight:** Phase 79 is mostly policy plumbing and conservative classification; custom runtime discovery or benchmark execution would expand into Phase 80/81 scope and weaken the claim boundary. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md]

## Common Pitfalls

### Pitfall 1: Treating Declared Tags As Validated Compatibility

**What goes wrong:** A manifest entry such as `7.2.3-complete` is reported as compatible because the tag exists. [CITED: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags]  
**Why it happens:** Tag existence proves a requested image tag is available, not that the host driver, device access, PyTorch stack, or benchmark execution validated it. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md]  
**How to avoid:** Manifest entries are requested Targets only; before evidence collection, status should be `not_tested` or `runtime_unavailable`, never `container_validated`. [VERIFIED: src/sol_execbench/core/compatibility.py]  
**Warning signs:** Tests assert `container_validated` from manifest parsing alone. [ASSUMED]

### Pitfall 2: Breaking The Current 7.1 Default

**What goes wrong:** Existing `./scripts/run_docker.sh --build` starts building a different ROCm image or requires a new flag. [VERIFIED: scripts/run_docker.sh]  
**Why it happens:** Parameterization removes the hardcoded default without reintroducing the exact same default in args and script selection. [VERIFIED: docker/Dockerfile]  
**How to avoid:** Add a default manifest entry and Dockerfile defaults for `rocm/dev-ubuntu-24.04:7.1.1-complete`; add a CPU-safe test that no-flag selection returns that exact tag. [VERIFIED: docker/Dockerfile] [ASSUMED]  
**Warning signs:** Dockerfile no longer contains `7.1.1-complete` as a default value, or no-target tests inspect a different Target id. [ASSUMED]

### Pitfall 3: Losing Matrix Semantics In Shell Exits

**What goes wrong:** Missing `/dev/dri` exits with a generic shell error and no structured `runtime_unavailable` evidence. [VERIFIED: scripts/run_docker.sh]  
**Why it happens:** Current `require_rocm_host_docker` prints errors and exits before constructing Matrix-compatible data. [VERIFIED: scripts/run_docker.sh]  
**How to avoid:** Have the script call a Python preflight mode that emits status/reason/build-arg data; shell can still stop execution after classification. [ASSUMED]  
**Warning signs:** Tests only grep stderr and never assert `MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE`. [ASSUMED]

### Pitfall 4: Over-Broad Unknown Target Override

**What goes wrong:** `--allow-unknown-target` becomes a normal path that can run full benchmarks or emit clean validation. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md]  
**Why it happens:** Override naming and decision flags are not tied to Phase 78 execution-decision semantics. [VERIFIED: src/sol_execbench/core/compatibility.py]  
**How to avoid:** Unknown overrides should produce `not_tested`, `benchmark_allowed=False`, `container_user_space_validated=False`, `native_host_validated=False`, and authority flags false. [VERIFIED: src/sol_execbench/core/compatibility.py]  
**Warning signs:** Unknown-target tests assert Docker build/run command construction without also asserting non-authoritative decision flags. [ASSUMED]

### Pitfall 5: Ignoring Docker Context Overrides

**What goes wrong:** A user is on `default`, but `DOCKER_HOST` or `DOCKER_CONTEXT` routes commands elsewhere. [CITED: https://docs.docker.com/engine/manage-resources/contexts/]  
**Why it happens:** Docker commands can be affected by current context, environment variables, and global flags. [CITED: https://docs.docker.com/engine/manage-resources/contexts/]  
**How to avoid:** Preserve `docker context show` plus `docker context inspect`, and record context name/host in preflight details. [VERIFIED: scripts/run_docker.sh] [ASSUMED]  
**Warning signs:** Classification records only "Docker Desktop" text without the actual context name/host. [ASSUMED]

## Code Examples

### Docker Build Args From Target

```bash
# Source: existing scripts/run_docker.sh build command plus Dockerfile ARG support
docker build \
  -t "${IMAGE}" \
  --build-arg "ROCM_DOCKER_IMAGE=${ROCM_DOCKER_IMAGE}" \
  --build-arg "ROCM_DOCKER_TAG=${ROCM_DOCKER_TAG}" \
  --build-arg HOST_UID="$(id -u)" \
  --build-arg HOST_GID="$(id -g)" \
  --build-arg HOST_USER="$(whoami)" \
  -f "${REPO_ROOT}/docker/Dockerfile" \
  "${REPO_ROOT}"
```

### Runtime-Unavailable Evidence Shape

```python
# Source: existing Phase 78 compatibility models
build_matrix_entry(
    target=target,
    observed=MatrixObservedEvidence(
        host=MatrixHostEvidence(
            device_nodes=observed_device_nodes,
            source="scripts/run_docker.sh preflight",
        ),
        container=MatrixContainerEvidence(
            image_repository=target.docker_image_repository,
            image_tag=target.docker_image_tag,
            image_digest=resolved_digest,
        ),
    ),
    status=MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE,
    reason_code=MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE,
    reason=reason,
    claim_boundary=MatrixClaimBoundary(
        container_user_space_validated=False,
        native_host_validated=False,
        hardware_validated=False,
    ),
)
```

### CPU-Safe Preflight Fixture

```python
# Source: tests/conftest.py path-exists injection style
def test_missing_dri_classifies_runtime_unavailable():
    observation = DockerPreflightObservation(
        context_name="default",
        docker_host="unix:///var/run/docker.sock",
        kfd_exists=True,
        dri_exists=False,
        gpu_probe_accessible=None,
    )

    result = classify_docker_preflight(observation)

    assert result.status is MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE
    assert result.benchmark_allowed is False
```

## State Of The Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single hardcoded ROCm Docker base image | Parameterized Docker base image selected from declared Targets | Phase 79 planned [VERIFIED: .planning/ROADMAP.md] | Enables auditable 7.0.x/7.1.x/7.2.x requested Matrix Targets while preserving default 7.1. [VERIFIED: .planning/REQUIREMENTS.md] |
| Shell exits for Docker preflight failures | Matrix-compatible `runtime_unavailable` classification before benchmark execution | Phase 79 planned [VERIFIED: .planning/ROADMAP.md] | Keeps setup failures distinct from benchmark correctness/performance. [VERIFIED: .planning/REQUIREMENTS.md] |
| Docker image tag as implicit environment | Target manifest with requested repository/tag/build args/digest evidence | Phase 79 planned [VERIFIED: .planning/ROADMAP.md] | Gives downstream Phase 81 enough data for per-Target reports. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md] |

**Deprecated/outdated:**
- Hardcoding only `FROM rocm/dev-ubuntu-24.04:7.1.1-complete` is insufficient for the v1.18 Docker Matrix, though it must remain the default. [VERIFIED: docker/Dockerfile] [VERIFIED: .planning/REQUIREMENTS.md]
- Treating Docker container validation as native host validation is explicitly forbidden by Phase 78. [VERIFIED: src/sol_execbench/core/compatibility.py]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Implement `src/sol_execbench/core/docker_matrix.py` plus `docker/rocm-targets.json`. | Summary / Architecture Patterns | Planner may choose different names; low risk if responsibilities stay the same. |
| A2 | Prefer `--target`, `--allow-unknown-target`, and `--preflight-only` flag names. | Summary / Standard Stack | User or planner may prefer env vars; tests need to follow final names. |
| A3 | JSON manifest is slightly simpler than TOML for strict fixture comparison and shell-adjacent tooling. | Standard Stack | TOML may be more readable; either is acceptable under user discretion. |
| A4 | Python preflight mode is the best way to keep shell behavior Matrix-compatible and CPU-testable. | Architecture Patterns / Pitfalls | If shell-only implementation is chosen, test coverage may need subprocess fixtures instead. |

## Open Questions (RESOLVED)

1. **RESOLVED: Which exact target ids should be public?**
   - What we know: Phase 79 needs configured 7.0.x, 7.1.x, and 7.2.x logical entries where usable image tags are known. [VERIFIED: .planning/REQUIREMENTS.md]
   - What's unclear: Whether ids should encode patch version and architecture, e.g. `rocm-7.1.1-container` versus `rocm-7.1-gfx1200-container`. [ASSUMED]
   - Recommendation: Use stable ids that include the logical ROCm family and validation scope; put exact patch tag in fields, not just the id. [ASSUMED]
   - Resolution: Plans use stable logical Target ids with ROCm family and container validation scope, while exact patch image tags remain manifest fields. [RESOLVED: .planning/phases/79-docker-matrix-selection-and-preflight/79-01-PLAN.md]

2. **RESOLVED: Which 7.2.x tag should be declared initially?**
   - What we know: Docker Hub currently lists `7.2-complete`, `7.2.1-complete`, `7.2.2-complete`, and `7.2.3-complete`. [CITED: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags]
   - What's unclear: The project has not pulled or validated these tags locally in this research turn. [VERIFIED: command output]
   - Recommendation: Declare the chosen tag as requested/auditable only, preferably the latest visible complete tag or a conservative patch selected by the user/planner, and do not claim live compatibility until Phase 81 evidence exists. [ASSUMED]
   - Resolution: Plan 79-01 delegates the exact patch value to the checked-in manifest and explicitly treats it as requested/auditable Target data only, never as a validation claim. [RESOLVED: .planning/phases/79-docker-matrix-selection-and-preflight/79-01-PLAN.md]

3. **RESOLVED: Should Phase 79 emit a JSON sidecar or only command-preview/preflight output?**
   - What we know: Full aggregate report emission is deferred to Phase 81, but Phase 79 must produce diagnostic Matrix-compatible evidence for preflight failures. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md]
   - What's unclear: Exact artifact path and CLI output contract for Phase 79. [ASSUMED]
   - Recommendation: Add a minimal `--preflight-json <path>` or stdout JSON preview for tests, without wiring full report generation. [ASSUMED]
   - Resolution: Plans require shell-consumable JSON/preview output from `sol_execbench.core.docker_matrix`, while full aggregate report emission remains deferred to Phase 81. [RESOLVED: .planning/phases/79-docker-matrix-selection-and-preflight/79-01-PLAN.md, .planning/phases/79-docker-matrix-selection-and-preflight/79-02-PLAN.md]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python / uv environment | Unit tests and helper module | yes | Python >=3.12 from project config; pytest 9.0.2 [VERIFIED: pyproject.toml] [VERIFIED: uv run pytest --version] | None needed. |
| Ruff | Lint | yes | 0.15.14 [VERIFIED: uv run ruff --version] | Use `uv run --with ruff ruff check ...` per AGENTS.md. [VERIFIED: AGENTS.md] |
| Docker CLI | Script build/run and digest/context checks | yes | 29.4.3 [VERIFIED: docker --version] | CPU-safe tests can mock/precompute command construction. [ASSUMED] |
| Docker daemon | Optional live Docker checks | yes | Server 29.4.3, context `default` [VERIFIED: docker info] | Phase planning should not require live build for CPU-safe tests. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md] |
| `/dev/kfd` | ROCm device passthrough | yes | char device present [VERIFIED: ls -l /dev/kfd] | Classify missing/inaccessible as `runtime_unavailable`. [VERIFIED: .planning/REQUIREMENTS.md] |
| `/dev/dri` | ROCm render-node passthrough | no | missing locally [VERIFIED: ls -ld /dev/dri] | Local live Docker run is currently blocked; CPU-safe tests should cover this as `runtime_unavailable`. [VERIFIED: .planning/REQUIREMENTS.md] |
| Native Linux Docker context | ROCm device passthrough | likely yes | context `default`; Docker daemon on Ubuntu 24.04 [VERIFIED: docker context show] [VERIFIED: docker info] | If Docker Desktop context is observed, classify as `runtime_unavailable`. [VERIFIED: scripts/run_docker.sh] |

**Missing dependencies with no fallback:**
- Live ROCm container benchmark execution is blocked in this environment because `/dev/dri` is absent, despite `/dev/kfd` being present. [VERIFIED: ls -ld /dev/dri] [VERIFIED: ls -l /dev/kfd]

**Missing dependencies with fallback:**
- Docker image digest resolution may be unavailable due to network/local image state; context requires digest capture to be best-effort and non-blocking. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Pytest 9.0.2 [VERIFIED: uv run pytest --version] |
| Config file | `pyproject.toml` with `addopts = "-n auto --dist loadgroup"` and ROCm markers. [VERIFIED: pyproject.toml] |
| Quick run command | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py -q` [ASSUMED] |
| Full suite command | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py -q` [ASSUMED] |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| DOCKER-01 | Manifest declares 7.0.x, 7.1.x default, and 7.2.x targets and converts to `MatrixTarget`. [VERIFIED: .planning/REQUIREMENTS.md] | unit/static | `uv run pytest tests/sol_execbench/test_docker_matrix_targets.py -q` [ASSUMED] | no, Wave 0 |
| DOCKER-02 | Dockerfile defaults to 7.1.1 and accepts `ROCM_DOCKER_IMAGE`/`ROCM_DOCKER_TAG` args before `FROM`. [VERIFIED: docker/Dockerfile] | static/unit | `uv run pytest tests/sol_execbench/test_docker_matrix_targets.py -q` [ASSUMED] | no, Wave 0 |
| DOCKER-03 | Script selection rejects unknown Targets unless explicit override is set; override remains non-authoritative. [VERIFIED: .planning/REQUIREMENTS.md] | unit/subprocess static | `uv run pytest tests/sol_execbench/test_docker_matrix_preflight.py -q` [ASSUMED] | no, Wave 0 |
| DOCKER-04 | Docker Desktop, missing `/dev/kfd`, missing `/dev/dri`, and GPU inaccessible classify as `runtime_unavailable`. [VERIFIED: .planning/REQUIREMENTS.md] | unit | `uv run pytest tests/sol_execbench/test_docker_matrix_preflight.py -q` [ASSUMED] | no, Wave 0 |
| DOCKER-05 | Requested repo/tag, optional digest, and build args are recorded in evidence/preview. [VERIFIED: .planning/REQUIREMENTS.md] | unit | `uv run pytest tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py -q` [ASSUMED] | no, Wave 0 |

### Sampling Rate

- **Per task commit:** `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py -q` [ASSUMED]
- **Per wave merge:** `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py -q` [ASSUMED]
- **Phase gate:** Add Ruff for touched Python files and optionally `bash -n scripts/run_docker.sh` plus static Dockerfile assertions; do not require live Docker/ROCm because `/dev/dri` may be absent in CPU-safe environments. [VERIFIED: AGENTS.md] [VERIFIED: ls -ld /dev/dri]

### Wave 0 Gaps

- [ ] `tests/sol_execbench/test_docker_matrix_targets.py` covers DOCKER-01, DOCKER-02, DOCKER-05. [ASSUMED]
- [ ] `tests/sol_execbench/test_docker_matrix_preflight.py` covers DOCKER-03, DOCKER-04, override boundaries. [ASSUMED]
- [ ] `src/sol_execbench/core/docker_matrix.py` provides pure helpers for tests and the shell wrapper. [ASSUMED]
- [ ] `docker/rocm-targets.json` or equivalent checked-in manifest. [ASSUMED]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No auth surface in this phase. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md] |
| V3 Session Management | no | No session state in this phase. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md] |
| V4 Access Control | yes | Keep Docker run access limited to required ROCm devices and existing project mount; do not add `--privileged`. [VERIFIED: scripts/run_docker.sh] [CITED: https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.0.0/how-to/docker.html] |
| V5 Input Validation | yes | Validate target ids and manifests with strict models; reject unknown Targets unless explicit override is supplied. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: src/sol_execbench/core/compatibility.py] |
| V6 Cryptography | no | No cryptographic operation in phase scope. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md] |

### Known Threat Patterns For This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Unknown image/tag injection into Docker build | Tampering | Only allow declared target ids by default; unknown override must be explicit and non-authoritative. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md] |
| Over-broad container privileges | Elevation of privilege | Preserve targeted `/dev/kfd`, `/dev/dri`, group, seccomp, IPC, and ulimit behavior; do not add broad host mounts or secrets. [VERIFIED: scripts/run_docker.sh] |
| Misleading compatibility claims | Spoofing / Repudiation | Use Phase 78 claim boundaries and status validation; container entries cannot claim native host validation. [VERIFIED: src/sol_execbench/core/compatibility.py] |
| Untrusted manifest edits | Tampering | Strict schema tests and bounded target ids; no runtime discovery of arbitrary tags as supported targets. [VERIFIED: .planning/REQUIREMENTS.md] [ASSUMED] |

## Sources

### Primary (HIGH confidence)

- `AGENTS.md` - project structure, commands, style, testing, security, and GSD workflow directives. [VERIFIED: AGENTS.md]
- `.planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md` - locked decisions, boundaries, and deferred scope. [VERIFIED: .planning/phases/79-docker-matrix-selection-and-preflight/79-CONTEXT.md]
- `.planning/REQUIREMENTS.md` - DOCKER-01 through DOCKER-05 and related traceability. [VERIFIED: .planning/REQUIREMENTS.md]
- `.planning/ROADMAP.md` and `.planning/STATE.md` - phase ordering and current focus. [VERIFIED: .planning/ROADMAP.md] [VERIFIED: .planning/STATE.md]
- `src/sol_execbench/core/compatibility.py` - Phase 78 Matrix models, statuses, reason codes, claim boundaries, and execution classifier. [VERIFIED: src/sol_execbench/core/compatibility.py]
- `scripts/run_docker.sh` and `docker/Dockerfile` - current Docker wrapper and base image behavior. [VERIFIED: scripts/run_docker.sh] [VERIFIED: docker/Dockerfile]
- `tests/sol_execbench/test_rocm_compatibility_matrix.py` and `tests/sol_execbench/test_matrix_claim_guardrails.py` - existing CPU-safe matrix contract and guardrail tests. [VERIFIED: tests/sol_execbench/test_rocm_compatibility_matrix.py] [VERIFIED: tests/sol_execbench/test_matrix_claim_guardrails.py]
- Dockerfile reference - `ARG` before `FROM` and `FROM` variable support. [CITED: https://docs.docker.com/reference/dockerfile]
- AMD ROCm Docker docs - host kernel sharing, `/dev/kfd`, `/dev/dri`, and `seccomp=unconfined` guidance. [CITED: https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.0.0/how-to/docker.html]

### Secondary (MEDIUM confidence)

- Docker contexts docs - active context, `DOCKER_CONTEXT`, `DOCKER_HOST`, and `docker context inspect` behavior. [CITED: https://docs.docker.com/engine/manage-resources/contexts/]
- Docker Hub `rocm/dev-ubuntu-24.04` tags page - currently visible 7.0.x/7.1.x/7.2.x tags and digests. [CITED: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags]

### Tertiary (LOW confidence)

- Recommendations for helper names, manifest filename, and exact flag spelling are local design assumptions under the user's discretion. [ASSUMED]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new packages; existing Python/Pydantic/Docker/Pytest/Ruff stack is verified locally. [VERIFIED: pyproject.toml] [VERIFIED: command output]
- Architecture: HIGH - existing file boundaries and Phase 78 contract are clear; helper names are assumed. [VERIFIED: scripts/run_docker.sh] [VERIFIED: src/sol_execbench/core/compatibility.py] [ASSUMED]
- Pitfalls: HIGH for claim-boundary and device-preflight risks; MEDIUM for Docker Hub tag currency. [VERIFIED: src/sol_execbench/core/compatibility.py] [CITED: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags]

**Research date:** 2026-05-28  
**Valid until:** 2026-06-04 for Docker Hub tag availability; 2026-06-27 for local architecture and tests.
