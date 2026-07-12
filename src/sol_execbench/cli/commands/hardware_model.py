# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Calibrated AMD hardware-model CLI artifacts.

These commands emit diagnostic evidence.  A successful file is not an
official score claim; the official gate independently requires exact validated
bound eligibility recorded on every score.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import click

from sol_execbench.core.scoring.amd_hardware_models import (
    AMD_HARDWARE_MODEL_V3_SCHEMA_VERSION,
)
from sol_execbench.core.scoring.hardware_calibration.builder import (
    CalibrationRequest,
    run_calibration,
)
from sol_execbench.core.scoring.hardware_calibration.environment import (
    adapter_for,
    discover_gpu,
)
from sol_execbench.core.scoring.hardware_calibration.models import (
    CALIBRATION_SCHEMA_VERSION,
    hardware_calibration_artifact_from_dict,
)
from sol_execbench.core.scoring.hardware_profile_requirements import (
    hardware_profile_requirements_from_dict,
)
from sol_execbench.core.scoring.hardware_calibration.rocprof_compute import (
    default_profiler_discovery,
    ensure_profiler_environment,
)
from sol_execbench.cli.protocol import (
    EXIT_UNAVAILABLE,
    CliFailure,
    CliResult,
    artifact,
)
from sol_execbench.cli.commands.fusion_validation import fusion_cli


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rejected(path: Path, reason: str) -> None:
    _write_json(
        path,
        {
            "schema_version": "sol_execbench.hardware_model_diagnostic.v1",
            "status": "rejected",
            "diagnostic": True,
            "reason": reason,
        },
    )


@click.group("hardware", context_settings={"help_option_names": ["-h", "--help"]})
def hardware_cli() -> None:
    """Manage hardware-derived benchmark evidence."""


hardware_cli.add_command(fusion_cli)


@hardware_cli.group("model")
def hardware_model_cli() -> None:
    """Create diagnostic calibration evidence and external hardware models."""


@hardware_model_cli.command("calibrate")
@click.option("--device", default=0, show_default=True, type=click.IntRange(min=0))
@click.option(
    "--output", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
@click.option(
    "--architecture", default=None, help="Assert the discovered AMD GFX architecture."
)
@click.option("--require-clock-lock", is_flag=True)
@click.option(
    "--requirements",
    "requirements_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Exact profile requirements derived from the score suite.",
)
@click.option("--offline", is_flag=True, help="Never install profiler dependencies.")
@click.option(
    "--no-auto-install", is_flag=True, help="Do not install profiler dependencies."
)
def _calibrate(
    device: int,
    output: Path,
    architecture: str | None,
    require_clock_lock: bool,
    requirements_path: Path | None,
    offline: bool,
    no_auto_install: bool,
) -> CliResult:
    """Collect calibration candidates without fabricating unavailable evidence."""
    try:
        environment = discover_gpu(device)
        if architecture and architecture.lower() != environment.architecture:
            raise ValueError(
                "architecture assertion does not match runtime discovery: "
                f"{architecture.lower()} != {environment.architecture}"
            )
        profiler = ensure_profiler_environment(
            default_profiler_discovery(Path.cwd()),
            offline=offline,
            auto_install=not no_auto_install,
        )
        requirements = _load_requirements(requirements_path, environment.architecture)
        artifact = run_calibration(
            CalibrationRequest(
                environment=environment,
                require_clock_lock=require_clock_lock,
                profiler_environment=profiler,
                required_profile_keys=(
                    requirements.required_profile_keys if requirements else ()
                ),
                requirements_ref=str(requirements_path) if requirements else None,
                requirements_sha256=(
                    requirements.payload_sha256 if requirements else None
                ),
            )
        )
    except (RuntimeError, ValueError) as exc:
        _rejected(output, str(exc))
        raise CliFailure(
            str(exc), code="environment_unavailable", exit_code=EXIT_UNAVAILABLE
        ) from exc
    if artifact.validation_status != "validated":
        reason = (
            "calibration is diagnostic only; validation provenance is not confirmed"
        )
        _rejected(output, reason)
        raise CliFailure(
            reason, code="environment_unavailable", exit_code=EXIT_UNAVAILABLE
        )
    _write_json(output, artifact.to_dict())
    return CliResult(
        data={"validation_status": artifact.validation_status},
        artifacts=(artifact_ref(output, "json_file"),),
    )


@hardware_model_cli.command("build")
@click.option(
    "--calibration",
    "calibration_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--verification-calibration",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Second independently collected calibration required with --requirements.",
)
@click.option(
    "--requirements",
    "requirements_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Exact profile requirements for this hardware model.",
)
@click.option(
    "--output", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
@click.option(
    "--max-age-hours",
    type=click.FloatRange(min=0.0),
    default=None,
    help="Reject calibration evidence older than this age.",
)
def _build(
    calibration_path: Path,
    output: Path,
    max_age_hours: float | None,
    verification_calibration: Path | None = None,
    requirements_path: Path | None = None,
) -> CliResult:
    """Convert a validated calibration artifact into an external v3 model."""
    try:
        raw_calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw_calibration, dict)
            or "payload_sha256" not in raw_calibration
        ):
            raise ValueError("calibration payload checksum is missing")
        calibration = hardware_calibration_artifact_from_dict(raw_calibration)
        architecture = str(calibration.metadata["architecture"])
        requirements = _load_requirements(requirements_path, architecture)
        if calibration.validation_status != "validated":
            raise ValueError("calibration validation status is not validated")
        required_profile_keys = (
            set(requirements.required_profile_keys)
            if requirements is not None
            else {candidate.key for candidate in calibration.candidates}
        )
        if any(
            candidate.state != "measured"
            for candidate in calibration.candidates
            if candidate.key in required_profile_keys
        ):
            raise ValueError("calibration contains non-measured hardware profiles")
        required_matrix_keys = {
            candidate.value
            for candidate in adapter_for(architecture).candidates
            if candidate.operation == "matrix"
            and candidate.input_dtype in {"bf16", "fp16"}
            and candidate.output_dtype == candidate.input_dtype
        }
        if requirements is not None:
            required_matrix_keys &= required_profile_keys
        missing_matrix_keys = required_matrix_keys - {
            candidate.key for candidate in calibration.candidates
        }
        if missing_matrix_keys:
            raise ValueError(
                "calibration missing required architecture matrix evidence: "
                + ", ".join(sorted(missing_matrix_keys))
            )
        _validate_calibration_authority(
            calibration,
            discover_gpu(int(calibration.metadata["device"])),
            max_age_hours,
        )
        verification = None
        if requirements is not None:
            if verification_calibration is None:
                raise ValueError(
                    "--verification-calibration is required with --requirements"
                )
            verification = _load_calibration(verification_calibration)
            _validate_second_calibration(
                primary=calibration,
                verification=verification,
                requirements_sha256=requirements.payload_sha256,
                architecture=architecture,
            )
            _validate_calibration_authority(
                verification,
                discover_gpu(int(verification.metadata["device"])),
                max_age_hours,
            )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        _rejected(output, str(exc))
        raise click.ClickException(str(exc)) from exc

    profiles = [
        {
            "key": candidate.key,
            "state": candidate.state,
            "value": _conservative_profile_value(candidate, verification),
            "confidence": "supported",
            "evidence_ref": (
                f"{calibration_path}#sha256:{calibration.payload_sha256}:"
                f"{candidate.key}"
                + (
                    f";{verification_calibration}#sha256:"
                    f"{verification.payload_sha256}:{candidate.key}"
                    if verification is not None
                    else ""
                )
            ),
        }
        for candidate in calibration.candidates
        if candidate.key in required_profile_keys
    ]
    _write_json(
        output,
        {
            "schema_version": AMD_HARDWARE_MODEL_V3_SCHEMA_VERSION,
            "architecture": architecture,
            "clock_assumptions": ["calibration provenance recorded in input artifact"],
            "source": f"calibrated from {calibration_path}",
            "confidence": "supported",
            "hardware_validation_status": "validated",
            "model_validation_status": "validated",
            "evidence_refs": [
                str(calibration_path),
                *([str(verification_calibration)] if verification is not None else []),
                *([str(requirements_path)] if requirements is not None else []),
            ],
            "compute_profiles": [
                p for p in profiles if str(p["key"]).startswith("compute.")
            ],
            "memory_profiles": [
                p for p in profiles if str(p["key"]).startswith("memory.")
            ],
        },
    )
    return CliResult(
        data={
            "schema_version": AMD_HARDWARE_MODEL_V3_SCHEMA_VERSION,
            "architecture": architecture,
        },
        artifacts=(artifact_ref(output, "json_file"),),
    )


def _validate_calibration_authority(
    calibration: object, live_environment: object, max_age_hours: float | None
) -> None:
    """Reject copied, stale, or locally inapplicable evidence before model output."""
    metadata = getattr(calibration, "metadata")
    required = (
        "gpu_uuid",
        "architecture",
        "rocm_version",
        "adapter_policy",
        "clock_observations",
    )
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ValueError(
            "calibration missing authority evidence: " + ", ".join(missing)
        )
    if metadata["gpu_uuid"] != getattr(live_environment, "uuid"):
        raise ValueError("calibration GPU UUID does not match current device")
    if metadata["architecture"] != getattr(live_environment, "architecture"):
        raise ValueError("calibration architecture does not match current device")
    if metadata["rocm_version"] != getattr(live_environment, "rocm_version"):
        raise ValueError("calibration ROCm version does not match current runtime")
    observations = metadata["clock_observations"]
    if (
        not isinstance(observations, dict)
        or observations.get("during") is not True
        or observations.get("post") is not False
    ):
        raise ValueError("calibration lacks observed clock lock/reset evidence")
    policy = metadata["adapter_policy"]
    if not isinstance(policy, dict) or policy.get("requires_clock_lock") is not True:
        raise ValueError("calibration adapter policy is not authority eligible")
    if max_age_hours is not None:
        try:
            generated = datetime.fromisoformat(
                cast(Any, calibration).generated_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("calibration timestamp is invalid") from exc
        if generated < datetime.now(UTC) - timedelta(hours=max_age_hours):
            raise ValueError("calibration evidence exceeds requested freshness window")


def _load_requirements(path: Path | None, architecture: str):
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("hardware profile requirements must be an object")
    requirements = hardware_profile_requirements_from_dict(payload)
    if requirements.architecture != architecture:
        raise ValueError("requirements architecture does not match calibration")
    return requirements


def _load_calibration(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "payload_sha256" not in payload:
        raise ValueError("calibration payload checksum is missing")
    return hardware_calibration_artifact_from_dict(payload)


def _validate_second_calibration(
    *,
    primary: object,
    verification: object,
    requirements_sha256: str | None,
    architecture: str,
) -> None:
    if getattr(verification, "schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("verification calibration has an unsupported schema")
    if getattr(verification, "validation_status") != "validated":
        raise ValueError("verification calibration validation status is not validated")
    metadata = getattr(verification, "metadata")
    if metadata.get("architecture") != architecture:
        raise ValueError("verification calibration architecture does not match")
    for calibration_artifact in (primary, verification):
        profile_requirements = getattr(calibration_artifact, "metadata").get(
            "profile_requirements"
        )
        if (
            not isinstance(profile_requirements, dict)
            or profile_requirements.get("requirements_sha256") != requirements_sha256
        ):
            raise ValueError("calibration does not bind the supplied requirements")
    first_time = getattr(primary, "generated_at")
    second_time = getattr(verification, "generated_at")
    if first_time == second_time:
        raise ValueError("calibrations must be independent runs")
    primary_by_key = {
        candidate.key: candidate for candidate in getattr(primary, "candidates")
    }
    verification_by_key = {
        candidate.key: candidate for candidate in getattr(verification, "candidates")
    }
    required_keys = getattr(primary, "metadata")["profile_requirements"][
        "required_profile_keys"
    ]
    for key in required_keys:
        left, right = primary_by_key.get(key), verification_by_key.get(key)
        if (
            left is None
            or right is None
            or left.state != "measured"
            or right.state != "measured"
        ):
            raise ValueError(f"independent calibration missing measured profile: {key}")
        assert left.value is not None and right.value is not None
        if abs(left.value - right.value) / min(left.value, right.value) > 0.05:
            raise ValueError(f"independent calibration differs by more than 5%: {key}")
        for calibration in (primary, verification):
            evidence = (
                getattr(calibration, "metadata").get("probe_evidence", {}).get(key)
            )
            required_evidence = {
                "compiler_command",
                "source_sha256",
                "binary_sha256",
                "warmup_iterations",
                "timed_samples",
            }
            if not isinstance(evidence, dict) or not required_evidence <= set(evidence):
                raise ValueError(f"calibration lacks probe provenance: {key}")


def _conservative_profile_value(
    candidate: object, verification: object | None
) -> float:
    value = getattr(candidate, "value")
    if value is None:
        raise ValueError("model cannot be built from an unmeasured profile")
    if verification is None:
        return float(value)
    other = next(
        entry
        for entry in getattr(verification, "candidates")
        if entry.key == getattr(candidate, "key")
    )
    if other.value is None:
        raise ValueError("verification calibration profile is unmeasured")
    return min(float(value), float(other.value))


artifact_ref = artifact
_hardware_model_cli = hardware_model_cli
