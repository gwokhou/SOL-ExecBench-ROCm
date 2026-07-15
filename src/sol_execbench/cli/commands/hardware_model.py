# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Calibrated AMD hardware-model CLI artifacts.

These commands emit diagnostic evidence.  A successful file is not an
official score claim; the official gate independently requires exact validated
bound eligibility recorded on every score.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import click

from sol_execbench.core.scoring.amd_hardware_models import (
    AMD_HARDWARE_MODEL_SCHEMA_VERSION,
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
from sol_execbench.core.scoring.hardware_calibration.shape_aware_roofline import (
    ShapeAwareRooflineArtifact,
    shape_aware_roofline_from_dict,
    validate_shape_aware_raw_evidence,
)
from sol_execbench.core.scoring.hardware_profile_requirements import (
    hardware_profile_requirements_from_dict,
)
from sol_execbench.core.integrity.checksums import sha256_file
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


def _validated_source_revision(source_revision: str | None) -> str | None:
    if (
        source_revision is not None
        and re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
    ):
        raise ValueError("source revision must be a full lowercase Git object id")
    return source_revision


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
@click.option(
    "--source-revision",
    help="Full Git revision of the source used for this calibration.",
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
    source_revision: str | None,
    offline: bool,
    no_auto_install: bool,
) -> CliResult:
    """Collect calibration candidates without fabricating unavailable evidence."""
    try:
        source_revision = _validated_source_revision(source_revision)
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
        calibration_artifact = run_calibration(
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
                source_revision=source_revision,
            )
        )
    except (RuntimeError, ValueError) as exc:
        _rejected(output, str(exc))
        raise CliFailure(
            str(exc), code="environment_unavailable", exit_code=EXIT_UNAVAILABLE
        ) from exc
    if calibration_artifact.validation_status != "validated":
        reason = (
            "calibration is diagnostic only; validation provenance is not confirmed"
        )
        _rejected(output, reason)
        raise CliFailure(
            reason, code="environment_unavailable", exit_code=EXIT_UNAVAILABLE
        )
    _write_json(output, calibration_artifact.to_dict())
    return CliResult(
        data={"validation_status": calibration_artifact.validation_status},
        artifacts=(artifact(output, "json_file"),),
    )


@hardware_model_cli.command("build")
@click.option(
    "--calibration",
    "calibration_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--shape-aware-evidence",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Checksummed shape/layout/launch/occupancy envelope evidence. Required "
        "only when the emitted model is intended to satisfy the authority envelope gate."
    ),
)
@click.option(
    "--authority-coverage",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Authority coverage artifact bound by --shape-aware-evidence.",
)
@click.option(
    "--shape-aware-plan",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Checksummed workload/profile plan that the envelope evidence must cover.",
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
    "--source-revision",
    help="Full Git revision that both calibration artifacts must bind.",
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
    shape_aware_evidence: Path | None = None,
    authority_coverage: Path | None = None,
    shape_aware_plan: Path | None = None,
    source_revision: str | None = None,
) -> CliResult:
    """Convert a validated calibration artifact into an external v3 model."""
    try:
        source_revision = _validated_source_revision(source_revision)
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
                source_revision=source_revision,
            )
            _validate_calibration_authority(
                verification,
                discover_gpu(int(verification.metadata["device"])),
                max_age_hours,
            )
        shape_evidence = _load_shape_aware_evidence(shape_aware_evidence)
        if shape_evidence is not None:
            if requirements is None or verification is None:
                raise ValueError(
                    "--shape-aware-evidence requires --requirements and --verification-calibration"
                )
            if authority_coverage is None:
                raise ValueError("--shape-aware-evidence requires --authority-coverage")
            if shape_aware_plan is None:
                raise ValueError("--shape-aware-evidence requires --shape-aware-plan")
            if shape_aware_evidence is None:
                raise ValueError("--shape-aware-evidence path is missing")
            primary_sha256 = calibration.payload_sha256
            verification_sha256 = verification.payload_sha256
            if primary_sha256 is None or verification_sha256 is None:
                raise ValueError("calibration payload checksum is missing")
            _validate_shape_aware_evidence(
                shape_evidence,
                evidence_path=shape_aware_evidence,
                plan=_load_shape_aware_plan(shape_aware_plan),
                architecture=architecture,
                primary_sha256=primary_sha256,
                verification_sha256=verification_sha256,
                requirements=requirements,
                authority_coverage_sha256=sha256_file(authority_coverage),
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
    model_payload: dict[str, object] = {
        "schema_version": AMD_HARDWARE_MODEL_SCHEMA_VERSION,
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
        "memory_profiles": [p for p in profiles if str(p["key"]).startswith("memory.")],
    }
    if shape_evidence is not None:
        model_payload["shape_aware_roofline"] = {
            "status": "validated",
            "evidence_refs": [
                f"{shape_aware_evidence}#sha256:{shape_evidence.payload_sha256}"
            ],
            "bucketing_dimensions": list(shape_evidence.bucketing_dimensions),
        }
    _write_json(output, model_payload)
    return CliResult(
        data={
            "schema_version": AMD_HARDWARE_MODEL_SCHEMA_VERSION,
            "architecture": architecture,
        },
        artifacts=(artifact(output, "json_file"),),
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
    initial_clock_state = (
        observations.get("pre") if isinstance(observations, dict) else None
    )
    if (
        not isinstance(observations, dict)
        or observations.get("during") is not True
        or (initial_clock_state is not True and initial_clock_state is not False)
        or observations.get("post") is not initial_clock_state
    ):
        raise ValueError("calibration lacks observed clock lock/restoration evidence")
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


def _load_shape_aware_evidence(
    path: Path | None,
) -> ShapeAwareRooflineArtifact | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shape-aware roofline evidence must be an object")
    return shape_aware_roofline_from_dict(payload)


def _load_shape_aware_plan(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "architecture",
        "authority_coverage_sha256",
        "requirements_sha256",
        "required_dimensions",
        "authority_workload_count",
        "profile_shards",
        "payload_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("shape-aware roofline plan has invalid fields")
    digest_payload = {
        key: value for key, value in payload.items() if key != "payload_sha256"
    }
    digest = hashlib.sha256(
        json.dumps(
            digest_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if payload["payload_sha256"] != digest:
        raise ValueError("shape-aware roofline plan checksum mismatch")
    return payload


def _validate_shape_aware_evidence(
    evidence: ShapeAwareRooflineArtifact,
    *,
    evidence_path: Path,
    plan: dict[str, object] | None = None,
    architecture: str,
    primary_sha256: str,
    verification_sha256: str,
    requirements: object,
    authority_coverage_sha256: str,
) -> None:
    """Bind a validated envelope to the same two scalar calibration runs.

    Envelope samples are diagnostic measurements, never inputs to the scalar
    peak/bandwidth values.  The only authority effect is satisfying the
    explicitly separate shape-aware evidence gate.
    """
    if evidence.architecture != architecture:
        raise ValueError("shape-aware roofline architecture does not match calibration")
    if set(evidence.calibration_sha256s) != {primary_sha256, verification_sha256}:
        raise ValueError("shape-aware roofline does not bind both calibration runs")
    if evidence.requirements_sha256 != getattr(requirements, "payload_sha256"):
        raise ValueError("shape-aware roofline does not bind supplied requirements")
    if evidence.authority_coverage_sha256 != authority_coverage_sha256:
        raise ValueError(
            "shape-aware roofline does not bind supplied authority coverage"
        )
    if plan is not None and evidence.plan_payload_sha256 != plan["payload_sha256"]:
        raise ValueError("shape-aware roofline does not bind supplied collection plan")
    required_keys = set(getattr(requirements, "required_profile_keys"))
    missing = required_keys - evidence.profile_keys
    if missing:
        raise ValueError(
            "shape-aware roofline lacks required profile evidence: "
            + ", ".join(sorted(missing))
        )
    for case in evidence.cases:
        raw_path = Path(case.raw_evidence_ref)
        if not raw_path.is_absolute():
            raw_path = evidence_path.parent / raw_path
        if not raw_path.is_file():
            raise ValueError(
                f"shape-aware roofline raw evidence is missing: {case.raw_evidence_ref}"
            )
        if sha256_file(raw_path) != case.raw_evidence_sha256:
            raise ValueError(
                f"shape-aware roofline raw evidence checksum mismatch: {case.raw_evidence_ref}"
            )
        if plan is not None:
            validate_shape_aware_raw_evidence(case, raw_path, architecture=architecture)
    if plan is None:
        return
    plan_assignments = _shape_aware_plan_assignments(plan)
    covered_assignments = [
        (case.profile_key, definition, workload_uuid, problem_id)
        for case in evidence.cases
        for definition, workload_uuid, problem_id in case.covered_workloads
    ]
    if len(covered_assignments) != len(set(covered_assignments)):
        raise ValueError("shape-aware roofline workload/profile coverage is duplicated")
    if set(covered_assignments) != plan_assignments:
        raise ValueError(
            "shape-aware roofline does not cover every planned workload/profile"
        )


def _shape_aware_plan_assignments(
    plan: dict[str, object],
) -> set[tuple[str, str, str, str]]:
    raw_shards = plan["profile_shards"]
    if not isinstance(raw_shards, list):
        raise ValueError("shape-aware roofline plan profile_shards must be a list")
    assignments: set[tuple[str, str, str, str]] = set()
    for raw_shard in raw_shards:
        if not isinstance(raw_shard, dict) or set(raw_shard) != {
            "profile_key",
            "workloads",
        }:
            raise ValueError("shape-aware roofline plan has an invalid profile shard")
        shard = cast(dict[str, object], raw_shard)
        profile_key = shard["profile_key"]
        raw_workloads = shard["workloads"]
        if not isinstance(profile_key, str) or not profile_key.strip():
            raise ValueError("shape-aware roofline plan profile_key must be non-empty")
        if not isinstance(raw_workloads, list):
            raise ValueError("shape-aware roofline plan workloads must be a list")
        for raw_workload in raw_workloads:
            if not isinstance(raw_workload, dict) or set(raw_workload) != {
                "definition",
                "workload_uuid",
                "problem_id",
            }:
                raise ValueError(
                    "shape-aware roofline plan has an invalid workload assignment"
                )
            workload = cast(dict[str, object], raw_workload)
            values = (
                workload["definition"],
                workload["workload_uuid"],
                workload["problem_id"],
            )
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(
                    "shape-aware roofline plan workload fields must be non-empty"
                )
            definition, workload_uuid, problem_id = cast(tuple[str, str, str], values)
            assignments.add((profile_key, definition, workload_uuid, problem_id))
    return assignments


def _validate_second_calibration(
    *,
    primary: object,
    verification: object,
    requirements_sha256: str | None,
    architecture: str,
    source_revision: str | None,
) -> None:
    if getattr(verification, "schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("verification calibration has an unsupported schema")
    if getattr(verification, "validation_status") != "validated":
        raise ValueError("verification calibration validation status is not validated")
    metadata = getattr(verification, "metadata")
    if metadata.get("architecture") != architecture:
        raise ValueError("verification calibration architecture does not match")
    for calibration_artifact in (primary, verification):
        metadata = getattr(calibration_artifact, "metadata")
        profile_requirements = getattr(calibration_artifact, "metadata").get(
            "profile_requirements"
        )
        if (
            not isinstance(profile_requirements, dict)
            or profile_requirements.get("requirements_sha256") != requirements_sha256
        ):
            raise ValueError("calibration does not bind the supplied requirements")
        if (
            source_revision is not None
            and metadata.get("source_revision") != source_revision
        ):
            raise ValueError("calibration does not bind the supplied source revision")
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
