# Phase 80: uv And PyTorch ROCm Wheel Coordination - Pattern Map

**Mapped:** 2026-05-28
**Files analyzed:** 8
**Analogs found:** 8 / 8

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/sol_execbench/core/dependency_matrix.py` | service/utility/model | request-response, transform | `src/sol_execbench/core/docker_matrix.py` + `src/sol_execbench/core/compatibility.py` | exact |
| `docker/rocm-targets.json` | config | request-response, transform | `docker/rocm-targets.json` | exact |
| `scripts/run_docker.sh` | route/script wrapper | request-response, process gating | `scripts/run_docker.sh` | exact |
| `tests/sol_execbench/test_dependency_matrix_policy.py` | test | config/schema validation | `tests/sol_execbench/test_docker_matrix_targets.py` | exact |
| `tests/sol_execbench/test_dependency_matrix_classification.py` | test | transform/classification | `tests/sol_execbench/test_matrix_claim_guardrails.py` + `tests/sol_execbench/test_docker_matrix_preflight.py` | exact |
| `tests/sol_execbench/test_dependency_matrix_cli.py` | test | subprocess request-response | `tests/sol_execbench/test_docker_matrix_preflight.py` | exact |
| `tests/sol_execbench/test_run_docker_dependency_preflight.py` | test | script subprocess gating | `tests/sol_execbench/test_run_docker_matrix_script.py` | exact |
| `pyproject.toml` | config | dependency resolution | `pyproject.toml` | exact |

## Pattern Assignments

### `src/sol_execbench/core/dependency_matrix.py` (service/utility/model, request-response + transform)

**Analog:** `src/sol_execbench/core/docker_matrix.py`

**Imports and strict model pattern** (`src/sol_execbench/core/docker_matrix.py` lines 3-28):
```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.compatibility import (
    MatrixClaimBoundary,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixContainerEvidence,
    MatrixEntry,
    MatrixExecutionDecision,
    MatrixGpuEvidence,
    MatrixHostEvidence,
    MatrixObservedEvidence,
    MatrixTarget,
    MatrixValidationScope,
    MatrixValidationScopeField,
    build_matrix_entry,
    classify_matrix_entry_for_execution,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings
```

**Strict Pydantic config pattern** (`src/sol_execbench/core/docker_matrix.py` lines 31-40):
```python
ROCM_DOCKER_TARGETS_SCHEMA_VERSION = "sol_execbench.rocm_docker_targets.v1"
DEFAULT_DOCKER_TARGET_MANIFEST = (
    Path(__file__).resolve().parents[3] / "docker" / "rocm-targets.json"
)
_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    use_attribute_docstrings=True,
)
```

**Manifest entry and validation pattern** (`src/sol_execbench/core/docker_matrix.py` lines 43-67):
```python
class DockerTargetManifestEntry(BaseModelWithDocstrings):
    """Declared Docker Target entry parsed from the repository manifest."""

    model_config = _MODEL_CONFIG

    target_id: str
    """Stable Target id for this Docker ROCm user-space request."""
    requested_rocm_user_space_version: str
    """Requested ROCm user-space version represented by the Docker image."""
    docker_image_repository: str
    """Requested Docker image repository."""
    docker_image_tag: str
    """Requested Docker image tag."""
    pytorch_rocm_target: str | None = None
    """Expected PyTorch ROCm wheel target for later dependency phases."""
    validation_scope: MatrixValidationScopeField
    """Validation scope for this Target."""
    intended_gpu_architecture: str | None = None
    """Intended AMD gfx architecture when the Target is architecture-specific."""

    @model_validator(mode="after")
    def _require_container_scope(self) -> DockerTargetManifestEntry:
        if self.validation_scope is not MatrixValidationScope.CONTAINER_USER_SPACE:
            raise ValueError("Docker Target manifest entries must use container_user_space")
        return self
```

**Load manifest pattern** (`src/sol_execbench/core/docker_matrix.py` lines 192-198):
```python
def load_docker_target_manifest(
    path: str | Path = DEFAULT_DOCKER_TARGET_MANIFEST,
) -> DockerTargetManifest:
    """Load and validate the checked-in Docker Target manifest."""

    payload = json.loads(Path(path).read_text())
    return DockerTargetManifest.model_validate(payload)
```

**Matrix target conversion pattern** (`src/sol_execbench/core/docker_matrix.py` lines 201-212):
```python
def to_matrix_target(target: DockerTargetManifestEntry) -> MatrixTarget:
    """Convert a declared Docker Target entry into the Phase 78 Matrix Target."""

    return MatrixTarget(
        target_id=target.target_id,
        requested_rocm_user_space_version=target.requested_rocm_user_space_version,
        docker_image_repository=target.docker_image_repository,
        docker_image_tag=target.docker_image_tag,
        pytorch_rocm_target=target.pytorch_rocm_target,
        validation_scope=target.validation_scope,
        intended_gpu_architecture=target.intended_gpu_architecture,
    )
```

**Classification-to-Matrix pattern** (`src/sol_execbench/core/docker_matrix.py` lines 371-424):
```python
def classify_docker_preflight(
    observation: DockerPreflightObservation,
) -> DockerPreflightResult:
    """Classify Docker runtime observations before benchmark execution."""

    reason = _runtime_unavailable_reason(observation)
    status = (
        MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE
        if reason is not None
        else MatrixCompatibilityStatus.NOT_TESTED
    )
    reason_code = (
        MatrixCompatibilityReasonCode.ROCM_RUNTIME_UNAVAILABLE
        if reason is not None
        else MatrixCompatibilityReasonCode.TARGET_NOT_TESTED
    )
    if reason is None:
        reason = (
            "Docker preflight did not find runtime blockers, but no benchmark, "
            "container user-space, host, or hardware validation has been "
            "performed."
        )
    target = to_matrix_target(observation.selected_target)
    entry = build_matrix_entry(
        target=target,
        observed=MatrixObservedEvidence(...),
        status=status,
        reason_code=reason_code,
        reason=reason,
        claim_boundary=MatrixClaimBoundary(
            container_user_space_validated=False,
            native_host_validated=False,
            hardware_validated=False,
        ),
    )
    return DockerPreflightResult(
        entry=entry,
        decision=classify_matrix_entry_for_execution(entry),
        build_args=dict(observation.build_args),
    )
```

**Shell-consumable JSON payload pattern** (`src/sol_execbench/core/docker_matrix.py` lines 162-189):
```python
def to_preview_payload(self) -> dict[str, Any]:
    """Return shell-consumable JSON for preflight classification."""

    entry_payload = self.entry.model_dump(mode="json")
    decision_payload = self.decision.model_dump(mode="json")
    target_payload = entry_payload["target"]
    container_payload = entry_payload["observed"]["container"]
    return {
        "target_id": target_payload["target_id"],
        "validation_scope": target_payload["validation_scope"],
        "status": entry_payload["status"],
        "reason_code": entry_payload["reason_code"],
        "reason": entry_payload["reason"],
        "benchmark_allowed": decision_payload["benchmark_allowed"],
        "probes_allowed": decision_payload["probes_allowed"],
        "smoke_allowed": decision_payload["smoke_allowed"],
        "score_authority": decision_payload["score_authority"],
        "paper_parity_authority": decision_payload["paper_parity_authority"],
        "leaderboard_authority": decision_payload["leaderboard_authority"],
        "container_user_space_validated": decision_payload[
            "container_user_space_validated"
        ],
        "native_host_validated": decision_payload["native_host_validated"],
    }
```

**CLI parser pattern** (`src/sol_execbench/core/docker_matrix.py` lines 469-540):
```python
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview = subparsers.add_parser("preview")
    preview.add_argument("--manifest", type=Path, default=DEFAULT_DOCKER_TARGET_MANIFEST)
    preview.add_argument("--target")
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--manifest", type=Path, default=DEFAULT_DOCKER_TARGET_MANIFEST)
    preflight.add_argument("--target")
    preflight.add_argument("--gpu-accessible", type=_parse_bool)
    return parser

def main(argv: list[str] | None = None) -> int:
    """Emit shell-consumable Docker Matrix JSON."""

    args = _build_parser().parse_args(argv)
    if args.command == "preview":
        payload = preview_docker_target_selection(...)
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.command == "preflight":
        payload = classify_docker_preflight(observation).to_preview_payload()
        print(json.dumps(payload, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")

if __name__ == "__main__":
    raise SystemExit(main())
```

**PyTorch runtime probe pattern** (`src/sol_execbench/core/environment.py` lines 440-475):
```python
def collect_pytorch_rocm_summary() -> PytorchRocmSummary:
    """Collect PyTorch ROCm metadata without requiring PyTorch at import time."""

    try:
        import torch
    except ImportError as exc:
        return PytorchRocmSummary(available=False, error=str(exc))

    torch_version = str(getattr(torch, "__version__", ""))
    version = getattr(torch, "version", None)
    hip_version = getattr(version, "hip", None)
    cuda_version = getattr(version, "cuda", None)
    try:
        available = bool(torch.cuda.is_available()) and hip_version is not None
        device_count = int(torch.cuda.device_count()) if available else 0
        device_name = torch.cuda.get_device_name(0) if device_count else None
        ...
```

Planner notes:
- Prefer a new `dependency_matrix.py` helper over bloating Bash.
- Use injectable `PytorchDependencyObservation` fixtures for most tests; a small collection function may use `importlib.metadata` plus the runtime probe style above.
- Build `MatrixObservedEvidence(python_dependency=MatrixPythonDependencyEvidence(...), toolchain=MatrixToolchainEvidence(...))`.
- For policy-unavailable wheels, set `MatrixCompatibilityStatus.PYTORCH_WHEEL_UNAVAILABLE` and `MatrixCompatibilityReasonCode.PYTORCH_ROCM_WHEEL_UNAVAILABLE`.
- For installed CPU/CUDA/wrong-ROCm/wrong-Triton/toolchain mismatches, set `MatrixCompatibilityStatus.MIXED_VERSION` and `MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH`.
- Call `classify_matrix_entry_for_execution(entry, allow_mixed_version_debug=...)` so debug override semantics stay centralized.

### `docker/rocm-targets.json` (config, request-response + transform)

**Analog:** `docker/rocm-targets.json`

**Current manifest shape** (`docker/rocm-targets.json` lines 1-33):
```json
{
  "schema_version": "sol_execbench.rocm_docker_targets.v1",
  "default_target_id": "rocm-7.1.1-ubuntu-24.04-container",
  "targets": [
    {
      "target_id": "rocm-7.0.2-ubuntu-24.04-container",
      "requested_rocm_user_space_version": "7.0.2",
      "docker_image_repository": "rocm/dev-ubuntu-24.04",
      "docker_image_tag": "7.0.2-complete",
      "pytorch_rocm_target": "rocm7.0",
      "validation_scope": "container_user_space",
      "intended_gpu_architecture": null
    },
    {
      "target_id": "rocm-7.1.1-ubuntu-24.04-container",
      "requested_rocm_user_space_version": "7.1.1",
      "docker_image_repository": "rocm/dev-ubuntu-24.04",
      "docker_image_tag": "7.1.1-complete",
      "pytorch_rocm_target": "rocm7.1",
      "validation_scope": "container_user_space",
      "intended_gpu_architecture": null
    }
  ]
}
```

Planner notes:
- Add a nested object near `pytorch_rocm_target`, for example `pytorch_dependency_policy`.
- Keep `default_target_id` and the 7.1.1 target unchanged.
- Record `torch_version`, `torchvision_version`, expected local tag such as `rocm7.1`, uv index name/url, lock strategy, and `triton-rocm` policy.
- Do not split policy into another manifest; phase decisions require target-adjacent dependency policy.

### `scripts/run_docker.sh` (route/script wrapper, request-response + process gating)

**Analog:** `scripts/run_docker.sh`

**Python JSON helper invocation pattern** (`scripts/run_docker.sh` lines 53-69):
```bash
resolve_docker_target_json() {
    local cmd=(
        python -m sol_execbench.core.docker_matrix preview
        --manifest "${REPO_ROOT}/docker/rocm-targets.json"
    )
    if [ -n "${DOCKER_TARGET}" ]; then
        cmd+=(--target "${DOCKER_TARGET}")
    fi
    if $ALLOW_UNKNOWN_TARGET; then
        cmd+=(
            --allow-unknown-target
            --override-image-repository "${ROCM_DOCKER_IMAGE:-rocm/dev-ubuntu-24.04}"
            --override-image-tag "${ROCM_DOCKER_TAG:-${DOCKER_TARGET}}"
        )
    fi
    PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${cmd[@]}"
}
```

**Preflight env override pattern** (`scripts/run_docker.sh` lines 71-79):
```bash
preflight_override_present() {
    [ -n "${SOL_EXECBENCH_DOCKER_CONTEXT:-}" ] ||
        [ -n "${SOL_EXECBENCH_DOCKER_HOST:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEV_KFD_PRESENT:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEV_KFD_ACCESSIBLE:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEV_DRI_PRESENT:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEV_DRI_ACCESSIBLE:-}" ] ||
        [ -n "${SOL_EXECBENCH_GPU_ACCESSIBLE:-}" ]
}
```

**Preflight classification command pattern** (`scripts/run_docker.sh` lines 108-140):
```bash
classify_docker_preflight_json() {
    local context_name
    local docker_host
    local dev_kfd_present
    local dev_kfd_accessible
    local dev_dri_present
    local dev_dri_accessible
    local cmd

    context_name="$(docker_context_name)"
    docker_host="$(docker_context_host)"
    dev_kfd_present="$(preflight_bool SOL_EXECBENCH_DEV_KFD_PRESENT "$(bool_text "$([ -e /dev/kfd ] && echo 1 || echo 0)")")"
    ...
    cmd=(
        python -m sol_execbench.core.docker_matrix preflight
        --manifest "${REPO_ROOT}/docker/rocm-targets.json"
        --docker-context "${context_name}"
        --docker-host "${docker_host}"
        --dev-kfd-present "${dev_kfd_present}"
        ...
    )
    if [ -n "${DOCKER_TARGET}" ]; then
        cmd+=(--target "${DOCKER_TARGET}")
    fi
    PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${cmd[@]}"
}
```

**Gate-before-build/run pattern** (`scripts/run_docker.sh` lines 197-214):
```bash
if [ "${DRY_RUN}" != "1" ] || $PREFLIGHT_ONLY || preflight_override_present; then
    PREFLIGHT_JSON="$(classify_docker_preflight_json)"
    PREFLIGHT_STATUS="$(matrix_json_value "${PREFLIGHT_JSON}" "status")"
    PREFLIGHT_BENCHMARK_ALLOWED="$(matrix_json_value "${PREFLIGHT_JSON}" "benchmark_allowed")"
    if $PREFLIGHT_ONLY; then
        echo "${PREFLIGHT_JSON}"
        if [ "${PREFLIGHT_STATUS}" = "runtime_unavailable" ] ||
            [ "${PREFLIGHT_BENCHMARK_ALLOWED}" != "True" ]; then
            exit 1
        fi
        exit 0
    fi
    if [ "${PREFLIGHT_STATUS}" = "runtime_unavailable" ] ||
        [ "${PREFLIGHT_BENCHMARK_ALLOWED}" != "True" ]; then
        echo "${PREFLIGHT_JSON}"
        exit 1
    fi
fi
```

Planner notes:
- Add an explicit dependency mismatch debug flag/env, for example `--allow-mixed-version-dependencies` and `SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES`, rather than reusing `--allow-unknown-target`.
- Add dependency preflight after target selection and before Docker build/run.
- Keep shell logic as JSON command assembly plus `status`/`benchmark_allowed` checks.
- Extend help text and tests so the explicit dependency override is visible and distinct from unknown-target override.

### `tests/sol_execbench/test_dependency_matrix_policy.py` (test, config/schema validation)

**Analog:** `tests/sol_execbench/test_docker_matrix_targets.py`

**Imports and constants pattern** (`tests/sol_execbench/test_docker_matrix_targets.py` lines 1-25):
```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sol_execbench.core.compatibility import (
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
    MatrixValidationScope,
)
from sol_execbench.core.docker_matrix import (
    docker_build_args_for_target,
    load_docker_target_manifest,
    preview_docker_target_selection,
    select_docker_target,
    to_matrix_target,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "docker" / "rocm-targets.json"
```

**Manifest default preservation pattern** (`tests/sol_execbench/test_docker_matrix_targets.py` lines 28-48):
```python
def test_manifest_declares_default_and_configured_rocm_complete_targets() -> None:
    manifest = load_docker_target_manifest(MANIFEST_PATH)

    assert manifest.default_target_id
    assert manifest.targets_by_id[manifest.default_target_id].docker_image_repository == (
        "rocm/dev-ubuntu-24.04"
    )
    assert manifest.targets_by_id[manifest.default_target_id].docker_image_tag == (
        "7.1.1-complete"
    )
    tags = {target.docker_image_tag for target in manifest.targets}
    assert any(tag.startswith("7.0.") and tag.endswith("-complete") for tag in tags)
    assert any(tag.startswith("7.2.") and tag.endswith("-complete") for tag in tags)
    assert {
        target.validation_scope for target in manifest.targets
    } == {MatrixValidationScope.CONTAINER_USER_SPACE}

    raw = MANIFEST_PATH.read_text()
    assert "rocm/dev-ubuntu-24.04" in raw
    assert "7.1.1-complete" in raw
```

**Preview JSON assertions pattern** (`tests/sol_execbench/test_docker_matrix_targets.py` lines 113-129):
```python
def test_default_preview_json_is_shell_consumable_without_docker() -> None:
    payload = preview_docker_target_selection(manifest_path=MANIFEST_PATH)

    assert payload["target_id"]
    assert payload["image_repository"] == "rocm/dev-ubuntu-24.04"
    assert payload["image_tag"] == "7.1.1-complete"
    assert payload["validation_scope"] == "container_user_space"
    assert payload["status"] == "not_tested"
    assert payload["reason_code"] == "target_not_tested"
    assert payload["benchmark_allowed"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False
```

Planner notes:
- Test that every declared target has dependency policy.
- Test that default target policy matches `torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, `triton-rocm==3.6.0`, and `pytorch-rocm71`.
- Test that `pyproject.toml` still contains the default ROCm 7.1 pins and uv source/index names.

### `tests/sol_execbench/test_dependency_matrix_classification.py` (test, transform/classification)

**Analogs:** `tests/sol_execbench/test_matrix_claim_guardrails.py`, `tests/sol_execbench/test_docker_matrix_preflight.py`

**Fixture builder pattern** (`tests/sol_execbench/test_matrix_claim_guardrails.py` lines 50-74):
```python
def _observed_container_stack(*, torch_rocm_target: str = "rocm7.1") -> MatrixObservedEvidence:
    return MatrixObservedEvidence(
        host=MatrixHostEvidence(
            rocm_version="7.1.0",
            driver_version="6.14.0",
            device_nodes=["/dev/kfd", "/dev/dri/renderD128"],
        ),
        container=MatrixContainerEvidence(
            rocm_user_space_version="7.1.0",
            image_repository="rocm/dev-ubuntu-24.04",
            image_tag="7.1.0-complete",
        ),
        python_dependency=MatrixPythonDependencyEvidence(
            python_version="3.12.10",
            torch_version=f"2.7.1+{torch_rocm_target}",
            torch_rocm_target=torch_rocm_target,
            torch_hip_version="7.1.0",
            triton_rocm_status="installed",
        ),
        gpu=MatrixGpuEvidence(
            device_count=1,
            device_name="AMD Radeon RX 9070 XT",
            gfx_architecture="gfx1200",
        ),
    )
```

**Mixed-version default block pattern** (`tests/sol_execbench/test_matrix_claim_guardrails.py` lines 136-153):
```python
def test_mixed_version_is_blocked_before_benchmark_by_default():
    entry = _entry(
        status=MatrixCompatibilityStatus.MIXED_VERSION,
        reason_code=MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH,
        reason="Target requested rocm7.1 but observed rocm7.0.",
    )

    decision = classify_matrix_entry_for_execution(
        entry, allow_mixed_version_debug=False
    )

    assert decision.status is MatrixCompatibilityStatus.MIXED_VERSION
    assert decision.reason_code is MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH
    assert decision.benchmark_allowed is False
    assert decision.probes_allowed is False
    assert decision.smoke_allowed is False
    assert "blocked before benchmark execution" in decision.reason
```

**Debug override authority pattern** (`tests/sol_execbench/test_matrix_claim_guardrails.py` lines 155-181):
```python
def test_mixed_version_debug_override_allows_probe_or_smoke_without_clean_claims():
    entry = _entry(
        status=MatrixCompatibilityStatus.MIXED_VERSION,
        reason_code=MatrixCompatibilityReasonCode.TARGET_OBSERVED_MISMATCH,
        reason="Observed PyTorch ROCm wheel target does not match requested Target.",
        claim_boundary=MatrixClaimBoundary(
            container_user_space_validated=True,
            native_host_validated=False,
            hardware_validated=True,
        ),
        observed=_observed_container_stack(torch_rocm_target="rocm7.0"),
    )

    decision = classify_matrix_entry_for_execution(
        entry, allow_mixed_version_debug=True
    )

    assert decision.benchmark_allowed is False
    assert decision.probes_allowed is True
    assert decision.smoke_allowed is True
    assert decision.container_user_space_validated is False
    assert decision.native_host_validated is False
    assert decision.score_authority is False
    assert decision.paper_parity_authority is False
    assert decision.leaderboard_authority is False
```

**Wheel unavailable pattern** (`tests/sol_execbench/test_matrix_claim_guardrails.py` lines 184-197):
```python
def test_pytorch_wheel_unavailable_is_not_a_benchmark_correctness_failure():
    entry = _entry(
        status=MatrixCompatibilityStatus.PYTORCH_WHEEL_UNAVAILABLE,
        reason_code=MatrixCompatibilityReasonCode.PYTORCH_ROCM_WHEEL_UNAVAILABLE,
        reason="No matching PyTorch ROCm wheel exists for this Target.",
    )

    decision = classify_matrix_entry_for_execution(entry)

    assert decision.benchmark_allowed is False
    assert decision.probes_allowed is True
    assert decision.smoke_allowed is False
    assert "dependency stack" in decision.reason
```

Planner notes:
- Add observations for missing `torch`, CPU wheel, CUDA wheel, wrong `+rocm` local tag, missing/wrong `triton-rocm`, and toolchain mismatch.
- Assert classifier output status and reason codes, not just reason strings.
- Always assert authority flags stay false for diagnostic blockers and debug overrides.

### `tests/sol_execbench/test_dependency_matrix_cli.py` (test, subprocess request-response)

**Analog:** `tests/sol_execbench/test_docker_matrix_preflight.py`

**Subprocess JSON pattern** (`tests/sol_execbench/test_docker_matrix_preflight.py` lines 126-168):
```python
def test_module_main_emits_preflight_json_from_explicit_observations() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sol_execbench.core.docker_matrix",
            "preflight",
            "--manifest",
            str(MANIFEST_PATH),
            "--docker-context",
            "desktop-linux",
            "--docker-host",
            "unix:///home/user/.docker/desktop/docker.sock",
            "--dev-kfd-present",
            "true",
            "--dev-kfd-accessible",
            "true",
            "--dev-dri-present",
            "true",
            "--dev-dri-accessible",
            "true",
            "--gpu-accessible",
            "false",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["target_id"]
    assert payload["validation_scope"] == "container_user_space"
    assert payload["status"] == "runtime_unavailable"
    assert payload["reason_code"] == "rocm_runtime_unavailable"
    assert payload["benchmark_allowed"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False
```

**Invalid argument pattern** (`tests/sol_execbench/test_docker_matrix_preflight.py` lines 171-202):
```python
def test_module_main_rejects_invalid_preflight_boolean_without_traceback() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sol_execbench.core.docker_matrix",
            "preflight",
            "--gpu-accessible",
            "maybe",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "expected boolean value" in completed.stderr
    assert "Traceback" not in completed.stderr
```

Planner notes:
- Mirror module invocation as `sys.executable, "-m", "sol_execbench.core.dependency_matrix", ...`.
- Use explicit CLI observation arguments or a JSON observation argument/env fixture so tests do not import live PyTorch or require hardware.
- Assert `status`, `reason_code`, `benchmark_allowed`, `probes_allowed`, `smoke_allowed`, and authority flags.

### `tests/sol_execbench/test_run_docker_dependency_preflight.py` (test, script subprocess gating)

**Analog:** `tests/sol_execbench/test_run_docker_matrix_script.py`

**Script subprocess helpers** (`tests/sol_execbench/test_run_docker_matrix_script.py` lines 48-81):
```python
def _run_docker_preview(*args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "SOL_EXECBENCH_RUN_DOCKER_DRY_RUN": "1",
    }
    return subprocess.run(
        [str(RUN_DOCKER_SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

def _run_docker_preflight(
    *args: str,
    **env_overrides: str,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "SOL_EXECBENCH_RUN_DOCKER_DRY_RUN": "1",
        **env_overrides,
    }
    return subprocess.run(
        [str(RUN_DOCKER_SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
```

**Preflight-only JSON blocking pattern** (`tests/sol_execbench/test_run_docker_matrix_script.py` lines 158-186):
```python
def test_run_docker_preflight_only_emits_runtime_unavailable_diagnostics() -> None:
    completed = _run_docker_preflight(
        "--preflight-only",
        SOL_EXECBENCH_DOCKER_CONTEXT="desktop-linux",
        SOL_EXECBENCH_DOCKER_HOST="unix:///home/user/.docker/desktop/docker.sock",
        SOL_EXECBENCH_DEV_KFD_PRESENT="true",
        SOL_EXECBENCH_DEV_KFD_ACCESSIBLE="true",
        SOL_EXECBENCH_DEV_DRI_PRESENT="true",
        SOL_EXECBENCH_DEV_DRI_ACCESSIBLE="true",
        SOL_EXECBENCH_GPU_ACCESSIBLE="false",
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "runtime_unavailable"
    assert payload["reason_code"] == "rocm_runtime_unavailable"
    assert payload["benchmark_allowed"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout
```

**Normal run blocking pattern** (`tests/sol_execbench/test_run_docker_matrix_script.py` lines 232-250):
```python
def test_run_docker_not_tested_preflight_blocks_normal_run() -> None:
    completed = _run_docker_preflight(
        "--",
        "sol-execbench",
        "tests/sol_execbench/samples/rmsnorm",
        SOL_EXECBENCH_DOCKER_CONTEXT="default",
        SOL_EXECBENCH_DOCKER_HOST="unix:///var/run/docker.sock",
        SOL_EXECBENCH_DEV_KFD_PRESENT="true",
        SOL_EXECBENCH_DEV_KFD_ACCESSIBLE="true",
        SOL_EXECBENCH_DEV_DRI_PRESENT="true",
        SOL_EXECBENCH_DEV_DRI_ACCESSIBLE="true",
        SOL_EXECBENCH_GPU_ACCESSIBLE="true",
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "not_tested"
    assert payload["benchmark_allowed"] is False
    assert "docker run" not in completed.stdout
```

Planner notes:
- Add env overrides for dependency observations, for example installed torch version/local tag/hip/cuda/triton values.
- Assert a mixed-version dependency state blocks before `docker build` and `docker run`.
- Assert explicit dependency debug override uses a distinct flag/env and still emits `benchmark_allowed=false` with authority false.

### `pyproject.toml` (config, dependency resolution)

**Analog:** `pyproject.toml`

**Default dependency pins and uv explicit indexes** (`pyproject.toml` lines 11-23, 69-93):
```toml
dependencies = [
    "torch==2.10.0; sys_platform != 'linux' and sys_platform != 'win32'",
    "torch==2.10.0+rocm7.1; sys_platform == 'linux' or sys_platform == 'win32'",
    "torchvision==0.25.0; sys_platform != 'linux' and sys_platform != 'win32'",
    "torchvision==0.25.0+rocm7.1; sys_platform == 'linux' or sys_platform == 'win32'",
    "triton-rocm==3.6.0; sys_platform == 'linux'",
]

[[tool.uv.index]]
name = "pytorch-rocm71"
url = "https://download.pytorch.org/whl/rocm7.1"
explicit = true

[[tool.uv.index]]
name = "pytorch-rocm-root"
url = "https://download.pytorch.org/whl/"
explicit = true

[tool.uv.sources]
torch = [
  { index = "pytorch-rocm71", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-rocm71", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
triton-rocm = [
  { index = "pytorch-rocm-root", marker = "sys_platform == 'linux'" },
]
```

Planner notes:
- Phase 80 should not change default `pyproject.toml` pins unless deliberately adding metadata only.
- Add tests that assert this default ROCm 7.1 path remains present.

## Shared Patterns

### Strict Models
**Source:** `src/sol_execbench/core/docker_matrix.py` lines 35-40 and `src/sol_execbench/core/compatibility.py` lines 20-25  
**Apply to:** `dependency_matrix.py` policy, observation, and result models
```python
ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    use_attribute_docstrings=True,
)
```

### Matrix Status And Reason Vocabulary
**Source:** `src/sol_execbench/core/compatibility.py` lines 28-48  
**Apply to:** Dependency classifier statuses and tests
```python
class MatrixCompatibilityStatus(str, Enum):
    HOST_VALIDATED = "host_validated"
    CONTAINER_VALIDATED = "container_validated"
    MIXED_VERSION = "mixed_version"
    PYTORCH_WHEEL_UNAVAILABLE = "pytorch_wheel_unavailable"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    NOT_TESTED = "not_tested"

class MatrixCompatibilityReasonCode(str, Enum):
    TARGET_OBSERVED_MISMATCH = "target_observed_mismatch"
    PYTORCH_ROCM_WHEEL_UNAVAILABLE = "pytorch_rocm_wheel_unavailable"
    ROCM_RUNTIME_UNAVAILABLE = "rocm_runtime_unavailable"
    TARGET_NOT_TESTED = "target_not_tested"
```

### Python Dependency Evidence
**Source:** `src/sol_execbench/core/compatibility.py` lines 163-180  
**Apply to:** Dependency preflight Matrix entries
```python
class MatrixPythonDependencyEvidence(BaseModelWithDocstrings):
    python_version: str | None = None
    torch_version: str | None = None
    torch_rocm_target: str | None = None
    torch_hip_version: str | None = None
    torch_cuda_version: str | None = None
    triton_rocm_status: str | None = None
```

### Diagnostic Claim Boundaries
**Source:** `src/sol_execbench/core/compatibility.py` lines 229-247  
**Apply to:** All dependency preflight results
```python
class MatrixClaimBoundary(BaseModelWithDocstrings):
    diagnostic_compatibility_evidence: Literal[True] = True
    score_authority: Literal[False] = False
    paper_parity_authority: Literal[False] = False
    leaderboard_authority: Literal[False] = False
    container_user_space_validated: bool
    native_host_validated: bool
    hardware_validated: bool
```

### Execution Decision Semantics
**Source:** `src/sol_execbench/core/compatibility.py` lines 455-553  
**Apply to:** Dependency debug override and shell gating
```python
def classify_matrix_entry_for_execution(
    entry: MatrixEntry,
    *,
    allow_mixed_version_debug: bool = False,
) -> MatrixExecutionDecision:
    if status is MatrixCompatibilityStatus.MIXED_VERSION:
        if allow_mixed_version_debug:
            return MatrixExecutionDecision(
                benchmark_allowed=False,
                probes_allowed=True,
                smoke_allowed=True,
                container_user_space_validated=False,
                native_host_validated=False,
                ...
            )
        return MatrixExecutionDecision(
            benchmark_allowed=False,
            probes_allowed=False,
            smoke_allowed=False,
            ...
        )
```

### Shell JSON Extraction
**Source:** `scripts/run_docker.sh` lines 35-37  
**Apply to:** Reading dependency preflight JSON in Bash
```bash
matrix_json_value() {
    python -c 'import json, sys; data=json.loads(sys.argv[1]); value=data; [value := value[part] for part in sys.argv[2].split(".")]; print(value)' "$1" "$2"
}
```

## No Analog Found

None. All likely Phase 80 files have exact or close analogs in the Phase 78/79 compatibility and Docker Target implementation.

## Metadata

**Analog search scope:** `src/sol_execbench/core/`, `docker/`, `scripts/`, `tests/sol_execbench/`, `pyproject.toml`  
**Files scanned:** 14  
**Pattern extraction date:** 2026-05-28
