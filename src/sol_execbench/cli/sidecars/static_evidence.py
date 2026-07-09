# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Static kernel evidence helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ...core.bench.static_kernel.evidence import (
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceSourceReference,
    StaticKernelEvidenceWarning,
    build_static_kernel_evidence_failed,
    build_static_kernel_evidence_unsupported,
    collect_static_kernel_artifacts,
    run_static_kernel_extractors,
)
from ...core.evidence.runtime_evidence import write_json_payload

console = Console(stderr=True)

STATIC_EVIDENCE_NONE = "none"
STATIC_EVIDENCE_AUTO = "auto"


def _static_evidence_directory(output_file: Path | None, staging_dir: Path) -> Path:
    """Return the static evidence artifact directory for an evaluation run."""

    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.static-evidence")
    return staging_dir / "static-evidence"


def _static_evidence_sidecar_path(output_file: Path | None, staging_dir: Path) -> Path:
    """Return the static evidence JSON sidecar path for an evaluation run."""

    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.static-evidence.json")
    return staging_dir / "static-evidence.json"


def _static_evidence_summary(
    sidecar: StaticKernelEvidenceSidecar,
) -> dict[str, object]:
    """Return a compact human-facing static evidence summary."""

    classification = sidecar.classification
    return {
        "status": sidecar.status,
        "reason_code": sidecar.reason_code,
        "artifact_count": len(sidecar.artifacts),
        "tool_run_count": len(sidecar.tool_runs),
        "metadata_present": classification.metadata_present,
        "disassembly_present": classification.disassembly_present,
        "detected_architectures": classification.detected_architectures,
        "unsupported_count": sum(
            1 for run in sidecar.tool_runs if run.status == "unsupported"
        ),
        "failed_count": sum(1 for run in sidecar.tool_runs if run.status == "failed"),
        "claim_boundaries": {
            "diagnostic_only": sidecar.diagnostic_only,
            "correctness_authority": sidecar.correctness_authority,
            "performance_authority": sidecar.performance_authority,
            "timing_authority": sidecar.timing_authority,
            "score_authority": sidecar.score_authority,
            "paper_parity_authority": sidecar.paper_parity_authority,
            "leaderboard_authority": sidecar.leaderboard_authority,
        },
    }


def _static_evidence_payload(
    sidecar: StaticKernelEvidenceSidecar,
) -> dict[str, object]:
    """Return the JSON sidecar payload with a compact summary section."""

    payload = sidecar.model_dump(mode="json")
    payload["summary"] = _static_evidence_summary(sidecar)
    return payload


def _write_static_evidence_sidecar(
    output_file: Path | None,
    staging_dir: Path,
    sidecar: StaticKernelEvidenceSidecar | None,
) -> Path | None:
    """Write optional static evidence metadata without changing trace JSONL."""

    if sidecar is None:
        return None

    sidecar_path = _static_evidence_sidecar_path(output_file, staging_dir)
    try:
        write_json_payload(sidecar_path, _static_evidence_payload(sidecar))
        console.print(
            "[green]Static evidence "
            f"{sidecar.status.value}; saved metadata to {sidecar_path}[/green]"
        )
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Static evidence metadata skipped: {exc}[/yellow]")
        return None


def _collect_static_evidence_for_cli(
    *,
    enabled: str,
    is_cpp: bool,
    staging_dir: Path,
    output_file: Path | None,
    target_architecture: str | None = None,
    artifact_collector=collect_static_kernel_artifacts,
    extractor_runner=run_static_kernel_extractors,
) -> StaticKernelEvidenceSidecar | None:
    """Collect optional static evidence for the CLI."""

    if enabled == STATIC_EVIDENCE_NONE:
        return None
    evidence_dir = _static_evidence_directory(output_file, staging_dir)
    if not is_cpp:
        return build_static_kernel_evidence_unsupported(
            StaticKernelEvidenceReasonCode.UNSUPPORTED_SOLUTION_TYPE
        )
    try:
        artifact_sidecar = artifact_collector(
            build_directory=staging_dir,
            evidence_directory=evidence_dir,
            sidecar_base_directory=evidence_dir,
            target_architecture=target_architecture,
        )
        if not artifact_sidecar.artifacts:
            return artifact_sidecar
        return extractor_runner(
            artifacts=artifact_sidecar.artifacts,
            evidence_directory=evidence_dir,
            sidecar_base_directory=evidence_dir,
        )
    except Exception as exc:
        sidecar = build_static_kernel_evidence_failed()
        return sidecar.model_copy(
            update={
                "warnings": [
                    StaticKernelEvidenceWarning(
                        code="static_evidence_collection_failed",
                        message=str(exc),
                        source_reference=StaticKernelEvidenceSourceReference(
                            kind="exception",
                            value=type(exc).__name__,
                            description="Static evidence collection is nonfatal.",
                        ),
                    )
                ]
            }
        )
