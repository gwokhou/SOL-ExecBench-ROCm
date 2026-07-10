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

import click

from sol_execbench.core.scoring.amd_hardware_models import (
    AMD_HARDWARE_MODEL_V3_SCHEMA_VERSION,
)
from sol_execbench.core.scoring.hardware_calibration.builder import (
    CalibrationRequest,
    run_calibration,
)
from sol_execbench.core.scoring.hardware_calibration.environment import discover_gpu
from sol_execbench.core.scoring.hardware_calibration.models import (
    hardware_calibration_artifact_from_dict,
)
from sol_execbench.core.scoring.hardware_calibration.rocprof_compute import (
    default_profiler_discovery,
    ensure_profiler_environment,
)


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


@click.group("hardware-model", context_settings={"help_option_names": ["-h", "--help"]})
def _hardware_model_cli() -> None:
    """Create diagnostic calibration evidence and external hardware models."""


@_hardware_model_cli.command("calibrate")
@click.option("--device", default=0, show_default=True, type=click.IntRange(min=0))
@click.option(
    "--output", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
@click.option(
    "--architecture", default=None, help="Assert the discovered AMD GFX architecture."
)
@click.option("--require-clock-lock", is_flag=True)
@click.option("--offline", is_flag=True, help="Never install profiler dependencies.")
@click.option(
    "--no-auto-install", is_flag=True, help="Do not install profiler dependencies."
)
def _calibrate(
    device: int,
    output: Path,
    architecture: str | None,
    require_clock_lock: bool,
    offline: bool,
    no_auto_install: bool,
) -> None:
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
        artifact = run_calibration(
            CalibrationRequest(
                environment=environment,
                require_clock_lock=require_clock_lock,
                profiler_environment=profiler,
            )
        )
    except (RuntimeError, ValueError) as exc:
        _rejected(output, str(exc))
        raise click.ClickException(str(exc)) from exc
    if artifact.validation_status != "validated":
        reason = (
            "calibration is diagnostic only; validation provenance is not confirmed"
        )
        _rejected(output, reason)
        raise click.ClickException(reason)
    _write_json(output, artifact.to_dict())


@_hardware_model_cli.command("build")
@click.option(
    "--calibration",
    "calibration_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
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
def _build(calibration_path: Path, output: Path, max_age_hours: float | None) -> None:
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
        if calibration.validation_status != "validated":
            raise ValueError("calibration validation status is not validated")
        if any(candidate.state != "measured" for candidate in calibration.candidates):
            raise ValueError("calibration contains non-measured hardware profiles")
        _validate_calibration_authority(
            calibration,
            discover_gpu(int(calibration.metadata["device"])),
            max_age_hours,
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        _rejected(output, str(exc))
        raise click.ClickException(str(exc)) from exc

    profiles = [
        {
            "key": candidate.key,
            "state": candidate.state,
            "value": candidate.value,
            "confidence": "supported",
            "evidence_ref": f"{calibration_path}#{candidate.key}",
        }
        for candidate in calibration.candidates
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
            "evidence_refs": [str(calibration_path)],
            "compute_profiles": [
                p for p in profiles if p["key"].startswith("compute.")
            ],
            "memory_profiles": [p for p in profiles if p["key"].startswith("memory.")],
        },
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
                calibration.generated_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("calibration timestamp is invalid") from exc
        if generated < datetime.now(UTC) - timedelta(hours=max_age_hours):
            raise ValueError("calibration evidence exceeds requested freshness window")
