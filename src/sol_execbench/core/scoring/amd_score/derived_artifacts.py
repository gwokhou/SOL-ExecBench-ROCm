"""Derived artifact helpers for AMD-native score report construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sol_execbench.core.evidence.evidence_refs import sidecar_stem_for_workload
from sol_execbench.core.scoring.amd_score.sidecar_parsing import (
    amd_sol_bound_from_payload,
    minimal_solar_aggregate_from_payload,
    read_json_object,
)
from sol_execbench.core.scoring.amd_hardware_models import (
    AmdHardwareModel,
    load_amd_hardware_model,
)
from sol_execbench.core.scoring.amd_sol import (
    AmdSolBoundArtifact,
    build_amd_sol_bound_artifact,
)
from sol_execbench.core.scoring.fusion_validation import FusionValidationArtifact
from sol_execbench.core.scoring.solar_derivation import (
    build_solar_derivation_evidence,
    solar_derivation_from_dict,
)


@dataclass(frozen=True)
class ResolvedHardwareModel:
    """Resolved external hardware model for derived score artifacts."""

    key: str
    model: AmdHardwareModel
    ref: str


@dataclass(frozen=True)
class DerivedScoreArtifacts:
    """Derived artifact inputs for one AMD-native workload score."""

    sol_bound_artifact: AmdSolBoundArtifact | None
    sol_bound_ref: str
    solar_derivation: Any | None
    evidence_refs: dict[str, str] | None


def resolve_hardware_model_from_path(path: Path) -> ResolvedHardwareModel:
    """Load a validated external model instead of silently using a package default."""
    path = Path(path).resolve()
    model = load_amd_hardware_model(path)
    return ResolvedHardwareModel(
        key=model.architecture,
        model=model,
        ref=str(path),
    )


def resolve_derived_score_artifacts(
    *,
    definition: Any,
    workload: Any | None,
    workload_uuid: str,
    hardware_model: ResolvedHardwareModel,
    fusion_validation: FusionValidationArtifact | None,
    fusion_validation_ref: str | None,
    fusion_validation_sha256: str | None,
    fusion_validation_path: Path | None,
    sol_bound_artifact_dir: Path | None,
    solar_derivation_dir: Path | None,
    sidecar_namespace: str | None,
    derived_exclusion: str | None,
) -> DerivedScoreArtifacts:
    """Resolve or generate derived sidecar artifacts for one score."""
    sidecar_stem = (
        sidecar_stem_for_workload(
            definition.name,
            workload_uuid,
            problem_namespace=sidecar_namespace,
        )
        if workload is not None
        else None
    )
    sol_bound_ref = f"derived:{definition.name}:{workload_uuid}:amd_sol_bound"
    sol_bound_path = (
        sol_bound_artifact_dir / f"{sidecar_stem}.amd-sol.json"
        if sol_bound_artifact_dir is not None and sidecar_stem is not None
        else None
    )
    sol_bound_artifact = _resolve_sol_bound_artifact(
        definition=definition,
        workload=workload,
        hardware_model=hardware_model,
        derived_exclusion=derived_exclusion,
        sol_bound_path=sol_bound_path,
        fusion_validation=fusion_validation,
        fusion_validation_ref=fusion_validation_ref,
        fusion_validation_sha256=fusion_validation_sha256,
        fusion_validation_path=fusion_validation_path,
    )
    if sol_bound_path is not None and sol_bound_artifact is not None:
        sol_bound_artifact_dir = sol_bound_path.parent
        sol_bound_artifact_dir.mkdir(parents=True, exist_ok=True)
        if not sol_bound_path.exists():
            sol_bound_path.write_text(
                json.dumps(sol_bound_artifact.to_dict(), indent=2)
            )
        sol_bound_ref = str(sol_bound_path)

    solar_derivation, evidence_refs = _resolve_solar_derivation(
        definition=definition,
        workload=workload,
        workload_uuid=workload_uuid,
        hardware_model=hardware_model,
        derived_exclusion=derived_exclusion,
        solar_derivation_dir=solar_derivation_dir,
        sidecar_stem=sidecar_stem,
    )
    return DerivedScoreArtifacts(
        sol_bound_artifact=sol_bound_artifact,
        sol_bound_ref=sol_bound_ref,
        solar_derivation=solar_derivation,
        evidence_refs=evidence_refs,
    )


def _resolve_sol_bound_artifact(
    *,
    definition: Any,
    workload: Any | None,
    hardware_model: ResolvedHardwareModel,
    derived_exclusion: str | None,
    sol_bound_path: Path | None,
    fusion_validation: FusionValidationArtifact | None,
    fusion_validation_ref: str | None,
    fusion_validation_sha256: str | None,
    fusion_validation_path: Path | None,
) -> AmdSolBoundArtifact | None:
    artifact = None
    if sol_bound_path is not None and sol_bound_path.exists():
        existing_payload = read_json_object(sol_bound_path)
        if existing_payload is not None:
            artifact = amd_sol_bound_from_payload(existing_payload)
    if artifact is None and workload is not None and derived_exclusion is None:
        if (
            fusion_validation is None
            or fusion_validation_ref is None
            or fusion_validation_sha256 is None
        ):
            raise ValueError("AMD SOL generation requires fusion-validation evidence")
        artifact = build_amd_sol_bound_artifact(
            definition,
            workload,
            hardware_model.model,
            fusion_validation=fusion_validation,
            fusion_validation_ref=fusion_validation_ref,
            fusion_validation_sha256=fusion_validation_sha256,
            evidence_path=fusion_validation_path,
            hardware_model_ref=hardware_model.ref,
        )
    return artifact


def _resolve_solar_derivation(
    *,
    definition: Any,
    workload: Any | None,
    workload_uuid: str,
    hardware_model: ResolvedHardwareModel,
    derived_exclusion: str | None,
    solar_derivation_dir: Path | None,
    sidecar_stem: str | None,
) -> tuple[Any | None, dict[str, str] | None]:
    solar_derivation = None
    evidence_refs = None
    solar_derivation_ref = f"derived:{definition.name}:{workload_uuid}:solar_derivation"
    solar_derivation_path = (
        solar_derivation_dir / f"{sidecar_stem}.solar-derivation.json"
        if solar_derivation_dir is not None and sidecar_stem is not None
        else None
    )
    if solar_derivation_path is not None and solar_derivation_path.exists():
        existing_payload = read_json_object(solar_derivation_path)
        if existing_payload is not None:
            solar_derivation = minimal_solar_aggregate_from_payload(existing_payload)
            if solar_derivation is not None:
                solar_derivation_ref = str(solar_derivation_path)
    if (
        workload is not None
        and solar_derivation_dir is not None
        and derived_exclusion is None
    ):
        solar_derivation_dir.mkdir(parents=True, exist_ok=True)
        assert solar_derivation_path is not None
        if solar_derivation is None:
            generated = build_solar_derivation_evidence(
                definition,
                workload,
                hardware_model.model,
                hardware_model_ref=hardware_model.ref,
            )
            generated_payload = generated.to_dict()
            solar_derivation_path.write_text(json.dumps(generated_payload, indent=2))
            solar_derivation_ref = str(solar_derivation_path)
            try:
                solar_derivation = solar_derivation_from_dict(generated_payload)
            except ValueError as exc:
                solar_derivation = None
                evidence_refs = {"solar_derivation_parse_error": str(exc)}
        else:
            solar_derivation_ref = str(solar_derivation_path)
        evidence_refs = {
            "formula": f"{solar_derivation_ref}#groups.formula_evidence",
            "hardware_model": hardware_model.ref,
            "coverage": f"{solar_derivation_ref}#coverage_summary",
            "score_eligibility": f"{solar_derivation_ref}#aggregate_status",
            **(evidence_refs or {}),
        }
    elif derived_exclusion is not None:
        evidence_refs = {
            "derived_sidecar_exclusion": derived_exclusion,
            "hardware_model": hardware_model.ref,
        }
    return solar_derivation, evidence_refs
