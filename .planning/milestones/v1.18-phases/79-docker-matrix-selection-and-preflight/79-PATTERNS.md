# Phase 79: Docker Matrix Selection And Preflight - Pattern Map

**Mapped:** 2026-05-28
**Files analyzed:** 6
**Analogs found:** 6 / 6

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/sol_execbench/core/docker_matrix.py` | service / model / utility | request-response, transform, file-I/O | `src/sol_execbench/core/compatibility.py` + `src/sol_execbench/core/environment.py` | exact |
| `docker/rocm-targets.json` | config | file-I/O, transform | `MatrixTarget` fields in `src/sol_execbench/core/compatibility.py` | role-match |
| `scripts/run_docker.sh` | route / wrapper script | request-response, command execution | `scripts/run_docker.sh` current preflight/build/run structure | exact |
| `docker/Dockerfile` | config | build-time transform | `docker/Dockerfile` current default ROCm image and host-user build args | exact |
| `tests/sol_execbench/test_docker_matrix_targets.py` | test | file-I/O, transform | `tests/sol_execbench/test_rocm_compatibility_matrix.py` + `tests/sol_execbench/test_environment_snapshot.py` | exact |
| `tests/sol_execbench/test_docker_matrix_preflight.py` | test | request-response, command/preflight classification | `tests/sol_execbench/test_matrix_claim_guardrails.py` + `tests/sol_execbench/test_rocm_marker_device_nodes.py` | exact |

## Pattern Assignments

### `src/sol_execbench/core/docker_matrix.py` (service / model / utility, request-response + transform + file-I/O)

**Analog:** `src/sol_execbench/core/compatibility.py` and `src/sol_execbench/core/environment.py`

**Imports and strict model pattern** (`src/sol_execbench/core/compatibility.py` lines 6-25):
```python
from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field, model_validator

from sol_execbench.core.data.base_model import BaseModelWithDocstrings


ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION = (
    "sol_execbench.rocm_compatibility_matrix.v1"
)
_MATRIX_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    use_attribute_docstrings=True,
)
```

Use the same `from __future__ import annotations`, Pydantic v2 models, `BaseModelWithDocstrings`, `extra="forbid"`, `frozen=True`, and explicit enum coercion where external JSON/string input is accepted.

**Target/evidence fields to convert into** (`src/sol_execbench/core/compatibility.py` lines 112-160):
```python
class MatrixTarget(BaseModelWithDocstrings):
    """Requested Target identity for a compatibility Matrix Entry."""

    model_config = _MATRIX_MODEL_CONFIG

    target_id: str
    requested_rocm_user_space_version: str
    docker_image_repository: str | None = None
    docker_image_tag: str | None = None
    pytorch_rocm_target: str | None = None
    validation_scope: MatrixValidationScopeField
    intended_gpu_architecture: str | None = None


class MatrixHostEvidence(BaseModelWithDocstrings):
    model_config = _MATRIX_MODEL_CONFIG
    device_nodes: list[str] = Field(default_factory=list)
    source: str | None = None


class MatrixContainerEvidence(BaseModelWithDocstrings):
    model_config = _MATRIX_MODEL_CONFIG
    image_repository: str | None = None
    image_tag: str | None = None
    image_digest: str | None = None
```

Manifest entries should be requested Targets only. Convert declared entries to `MatrixTarget`; record runtime observations separately in `MatrixHostEvidence` / `MatrixContainerEvidence`.

**Bounded preflight probe pattern** (`src/sol_execbench/core/environment.py` lines 177-187, 390-437):
```python
@dataclass(frozen=True)
class ProbeCompletedProcess:
    """Small subprocess result shape used by injectable probe runners."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


ProbeRunner = Callable[[list[str], float], ProbeCompletedProcess]
Which = Callable[[str], str | None]

def probe_tool(...):
    path = which(tool)
    if path is None:
        return ToolProbeResult(... status=EnvironmentEvidenceStatus.UNAVAILABLE)
    try:
        completed = runner(command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return ToolProbeResult(... status=EnvironmentEvidenceStatus.TIMEOUT)
    except OSError as exc:
        return ToolProbeResult(... status=EnvironmentEvidenceStatus.FAILED)
```

For Docker context/digest/probe helpers, keep command execution injectable and timeout-bounded. Tests should pass fake runners and filesystem observations, not require Docker or GPU devices.

**Matrix entry construction and execution decision pattern** (`src/sol_execbench/core/compatibility.py` lines 432-510):
```python
def build_matrix_entry(
    *,
    target: MatrixTarget,
    observed: MatrixObservedEvidence,
    status: MatrixCompatibilityStatus | str,
    reason_code: MatrixCompatibilityReasonCode | str,
    reason: str,
    claim_boundary: MatrixClaimBoundary,
    artifacts: Sequence[MatrixArtifactReference] = (),
) -> MatrixEntry:
    return MatrixEntry(
        target=target,
        observed=observed,
        status=status,
        reason_code=reason_code,
        reason=reason,
        claim_boundary=claim_boundary,
        artifacts=list(artifacts),
    )

def classify_matrix_entry_for_execution(...):
    if status is MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE:
        return MatrixExecutionDecision(
            benchmark_allowed=False,
            probes_allowed=True,
            smoke_allowed=False,
            container_user_space_validated=False,
            native_host_validated=False,
        )
```

Preflight failures should build `MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE` with `MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE`. Unknown/unsafe overrides should build `NOT_TESTED` / `TARGET_NOT_TESTED` and then use `classify_matrix_entry_for_execution` semantics for non-authoritative decisions.

---

### `docker/rocm-targets.json` (config, file-I/O + transform)

**Analog:** `MatrixTarget` in `src/sol_execbench/core/compatibility.py`

**Required schema fields by analog** (`src/sol_execbench/core/compatibility.py` lines 112-130):
```python
class MatrixTarget(BaseModelWithDocstrings):
    target_id: str
    requested_rocm_user_space_version: str
    docker_image_repository: str | None = None
    docker_image_tag: str | None = None
    pytorch_rocm_target: str | None = None
    validation_scope: MatrixValidationScopeField
    intended_gpu_architecture: str | None = None
```

Manifest JSON should include declared logical Targets for 7.0.x, 7.1.x default, and 7.2.x. Keep exact Docker repository/tag as data, not code. Include a default marker or default id so no-flag `run_docker.sh` keeps `rocm/dev-ubuntu-24.04:7.1.1-complete`.

**Validation vocabulary** (`src/sol_execbench/core/compatibility.py` lines 28-54):
```python
class MatrixCompatibilityStatus(str, Enum):
    HOST_VALIDATED = "host_validated"
    CONTAINER_VALIDATED = "container_validated"
    MIXED_VERSION = "mixed_version"
    PYTORCH_WHEEL_UNAVAILABLE = "pytorch_wheel_unavailable"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    NOT_TESTED = "not_tested"

class MatrixValidationScope(str, Enum):
    NATIVE_HOST = "native_host"
    CONTAINER_USER_SPACE = "container_user_space"
```

Do not add manifest status strings such as `docker_unavailable` or `unknown_target`; use existing Matrix status/reason vocabulary in Python.

---

### `scripts/run_docker.sh` (route / wrapper script, request-response + command execution)

**Analog:** existing `scripts/run_docker.sh`

**Shell entrypoint and argument-splitting pattern** (`scripts/run_docker.sh` lines 1-23, 64-83):
```bash
#!/bin/bash
# Launch the sol-execbench Docker container with the right mounts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-sol-execbench}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

BUILD=false
DOCKER_ARGS=()
CMD=()
seen_separator=false
for arg in "$@"; do
    if [ "$arg" = "--build" ] && ! $seen_separator; then
        BUILD=true
        continue
    fi
    if [ "$arg" = "--" ]; then
        seen_separator=true
        continue
    fi
    if $seen_separator; then
        CMD+=("$arg")
    else
        DOCKER_ARGS+=("$arg")
    fi
done
```

Extend this parser with `--target`, `--allow-unknown-target`, `--preflight-only`, and optional JSON/preview flags before the `--` separator. Keep existing no-flag behavior unchanged.

**Current host preflight checks to preserve, but classify through Python** (`scripts/run_docker.sh` lines 28-62):
```bash
context_name="$(docker context show 2>/dev/null || true)"
docker_host="$(docker context inspect --format '{{ (index .Endpoints "docker").Host }}' 2>/dev/null || true)"

if [[ "${context_name}" == "desktop-linux" || "${docker_host}" == *"/.docker/desktop/"* ]]; then
    ...
    exit 1
fi

if [ ! -e /dev/kfd ]; then
    echo "ERROR: /dev/kfd is missing on the host..." >&2
    exit 1
fi

if [ ! -d /dev/dri ]; then
    echo "ERROR: /dev/dri is missing on the host..." >&2
    exit 1
fi
```

Keep the same observations (`context_name`, `docker_host`, `/dev/kfd`, `/dev/dri`), but feed them into `docker_matrix.py` so failures produce Matrix-compatible `runtime_unavailable` evidence before exiting.

**Build args and run device pattern** (`scripts/run_docker.sh` lines 87-97, 116-136):
```bash
if $BUILD; then
    docker build \
        -t "${IMAGE}" \
        --build-arg HOST_UID="$(id -u)" \
        --build-arg HOST_GID="$(id -g)" \
        --build-arg HOST_USER="$(whoami)" \
        -f "${REPO_ROOT}/docker/Dockerfile" \
        "${REPO_ROOT}"
fi

DOCKER_CMD=(
    docker run --rm
    --device=/dev/kfd
    --device=/dev/dri
    --group-add video
    --security-opt seccomp=unconfined
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${REPO_ROOT}:${CONTAINER_PROJECT}"
    "${IMAGE}"
    "${CMD[@]}"
)
```

Add `--build-arg "ROCM_DOCKER_IMAGE=..."` and `--build-arg "ROCM_DOCKER_TAG=..."` from the selected Target. Preserve targeted ROCm device mounts and do not switch to broad `--privileged`.

---

### `docker/Dockerfile` (config, build-time transform)

**Analog:** existing `docker/Dockerfile`

**Default ROCm base image pattern to preserve** (`docker/Dockerfile` lines 1-11):
```dockerfile
# Adds ROCm runtime, tooling, and python dependencies. Used for benchmarking.

FROM rocm/dev-ubuntu-24.04:7.1.1-complete AS base

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib
```

Replace the hardcoded `FROM` with pre-`FROM` args whose defaults are exactly `rocm/dev-ubuntu-24.04` and `7.1.1-complete`; repeat args after `FROM` only if later instructions need them.

**Host-user build-arg pattern to preserve** (`docker/Dockerfile` lines 25-55):
```dockerfile
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG HOST_USER=sol-execbench

ENV HOME=/home/${HOST_USER} \
    UV_CACHE_DIR=/home/${HOST_USER}/.cache/uv

RUN if [ "${HOST_UID}" = "0" ]; then \
        RUN_USER=root; \
    else \
        ...
    fi && \
    mkdir -p /sol-execbench /venv /home/${HOST_USER}/.cache/uv && \
    chown -R ${HOST_UID}:${HOST_GID} /sol-execbench /venv /home/${HOST_USER}
```

Do not restructure dependency installation or user creation for this phase. The Dockerfile change should be narrowly scoped to ROCm base-image parameterization.

---

### `tests/sol_execbench/test_docker_matrix_targets.py` (test, file-I/O + transform)

**Analog:** `tests/sol_execbench/test_rocm_compatibility_matrix.py` and `tests/sol_execbench/test_environment_snapshot.py`

**Representative Matrix fixture pattern** (`tests/sol_execbench/test_rocm_compatibility_matrix.py` lines 36-93):
```python
def _representative_entry() -> MatrixEntry:
    return build_matrix_entry(
        target=MatrixTarget(
            target_id="rocm-7.1-gfx1200-container",
            requested_rocm_user_space_version="7.1.0",
            docker_image_repository="rocm/dev-ubuntu-24.04",
            docker_image_tag="7.1.0-complete",
            pytorch_rocm_target="rocm7.1",
            validation_scope=MatrixValidationScope.CONTAINER_USER_SPACE,
            intended_gpu_architecture="gfx1200",
        ),
        observed=MatrixObservedEvidence(...),
        status=MatrixCompatibilityStatus.CONTAINER_VALIDATED,
        reason_code=MatrixCompatibilityReasonCode.CONTAINER_USER_SPACE_VALIDATED,
        claim_boundary=MatrixClaimBoundary(...),
    )
```

Target tests should assert manifest entries convert to `MatrixTarget` with repository/tag/scope fields intact. Use the same model round-trip style, but do not assert manifest parsing alone produces `CONTAINER_VALIDATED`.

**Serialization and strictness pattern** (`tests/sol_execbench/test_rocm_compatibility_matrix.py` lines 96-134):
```python
payload = entry.model_dump(mode="json")

assert payload["target"]["docker_image_repository"] == "rocm/dev-ubuntu-24.04"
assert payload["target"]["docker_image_tag"] == "7.1.0-complete"
assert payload["target"]["validation_scope"] == "container_user_space"
assert MatrixEntry.model_validate(payload) == entry

with pytest.raises(ValidationError):
    MatrixEntry.model_validate(nested_payload)
```

Add tests for exact default target selection, unknown target rejection, explicit unknown override becoming non-authoritative, and Docker build arg construction.

**Injected runner / no live dependency pattern** (`tests/sol_execbench/test_environment_snapshot.py` lines 48-65, 122-159):
```python
calls: list[list[str]] = []

def runner(command: list[str], timeout_seconds: float) -> ProbeCompletedProcess:
    calls.append(command)
    return ProbeCompletedProcess(returncode=0)

result = probe_tool(
    "rocminfo",
    ["rocminfo"],
    runner=runner,
    which=lambda _tool: None,
    timeout_seconds=1.0,
)

assert result.status == EnvironmentEvidenceStatus.UNAVAILABLE
assert calls == []
```

For digest or Docker CLI lookup tests, use fake runners/which functions and assert missing digest is `None` / non-blocking.

---

### `tests/sol_execbench/test_docker_matrix_preflight.py` (test, request-response + classification)

**Analog:** `tests/sol_execbench/test_matrix_claim_guardrails.py` and `tests/sol_execbench/test_rocm_marker_device_nodes.py`

**Reusable target/evidence helper pattern** (`tests/sol_execbench/test_matrix_claim_guardrails.py` lines 28-118):
```python
def _container_target() -> MatrixTarget:
    return MatrixTarget(
        target_id="rocm-7.1-gfx1200-container",
        requested_rocm_user_space_version="7.1.0",
        docker_image_repository="rocm/dev-ubuntu-24.04",
        docker_image_tag="7.1.0-complete",
        pytorch_rocm_target="rocm7.1",
        validation_scope=MatrixValidationScope.CONTAINER_USER_SPACE,
        intended_gpu_architecture="gfx1200",
    )

def _entry(... ) -> MatrixEntry:
    return build_matrix_entry(
        target=_container_target(),
        observed=observed or _observed_container_stack(),
        status=status,
        reason_code=reason_code,
        reason=reason,
        claim_boundary=claim_boundary or MatrixClaimBoundary(
            container_user_space_validated=False,
            native_host_validated=False,
            hardware_validated=False,
        ),
    )
```

Use local helper constructors for Docker preflight observations and expected Matrix entries. Keep claim flags explicit.

**Runtime unavailable and not-tested assertions** (`tests/sol_execbench/test_matrix_claim_guardrails.py` lines 199-230):
```python
entry = _entry(
    status=MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE,
    reason_code=MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE,
    reason="Required ROCm runtime devices were unavailable.",
)

decision = classify_matrix_entry_for_execution(entry)

assert decision.benchmark_allowed is False
assert decision.probes_allowed is True
assert decision.smoke_allowed is False
assert "runtime" in decision.reason
```

Cover Docker Desktop context, missing `/dev/kfd`, missing `/dev/dri`, and inaccessible GPU device access as `RUNTIME_UNAVAILABLE`. Cover unknown override as `NOT_TESTED`, with `benchmark_allowed=False`, `container_user_space_validated=False`, and all authority flags false.

**Filesystem injection pattern for device nodes** (`tests/sol_execbench/test_rocm_marker_device_nodes.py` lines 17-39):
```python
available, gfx_arch, reason = conftest._rocm_gpu_info(
    path_exists=lambda _path: False
)

assert available is False
assert "/dev/kfd" in reason
assert "/dev/dri" in reason

missing = conftest._missing_rocm_device_nodes(
    path_exists=lambda path: path != Path("/dev/kfd")
)

assert missing == ("/dev/kfd",)
```

Preflight tests should inject host observations or path-exists callbacks; they should not depend on this workstation having `/dev/dri`.

## Shared Patterns

### Claim Boundaries
**Source:** `src/sol_execbench/core/compatibility.py` lines 278-362
**Apply to:** `docker_matrix.py`, target/preflight tests

Container-scoped Docker evidence must never use `host_validated` or `native_host_validated=true`. `container_validated` requires observed container evidence and `container_user_space_validated=true`; preflight failures and unknown overrides should keep clean validation flags false.

### Runtime-Unavailable Classification
**Source:** `src/sol_execbench/core/compatibility.py` lines 497-510 and `tests/sol_execbench/test_matrix_claim_guardrails.py` lines 199-211
**Apply to:** Docker Desktop context, missing `/dev/kfd`, missing `/dev/dri`, inaccessible GPU devices

Use `MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE` and `MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE`. This allows bounded diagnostic probes but blocks benchmark and smoke execution.

### Unknown/Untested Overrides
**Source:** `src/sol_execbench/core/compatibility.py` lines 455-510 and `tests/sol_execbench/test_matrix_claim_guardrails.py` lines 214-230
**Apply to:** unknown Docker Target override path

Use `MatrixCompatibilityStatus.NOT_TESTED` and `MatrixCompatibilityReasonCode.TARGET_NOT_TESTED`; assert no benchmark eligibility, no clean validation flags, and no score/paper/leaderboard authority.

### Docker Device Access
**Source:** `scripts/run_docker.sh` lines 28-62 and 116-136
**Apply to:** preflight and run command construction

Preserve Docker context detection, `/dev/kfd`, `/dev/dri`, `--group-add video`, `--security-opt seccomp=unconfined`, `--ipc=host`, and existing ulimits. Do not add broad privileges.

### CPU-Safe Tests
**Source:** `tests/sol_execbench/test_environment_snapshot.py` lines 48-65, 122-159 and `tests/sol_execbench/test_rocm_marker_device_nodes.py` lines 17-39
**Apply to:** all Phase 79 tests

Use injected runners, fake `which`, fake path existence, and structured observations. Avoid live Docker pulls/builds/runs and avoid requiring real ROCm device nodes.

## No Analog Found

All planned files have close analogs. There is no existing checked-in Docker Target manifest, so `docker/rocm-targets.json` should be modeled from `MatrixTarget` fields rather than copied from another config file.

## Metadata

**Analog search scope:** `src/sol_execbench/core/`, `scripts/`, `docker/`, `tests/sol_execbench/`, `tests/docker/dependencies/`
**Files scanned:** 14
**Pattern extraction date:** 2026-05-28
