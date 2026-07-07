# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic sidecar helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from ..core import Solution, Trace
from ..core.bench.agent_feedback import (
    AgentFeedbackArtifactCitation,
    artifact_citation_from_path,
    build_agent_feedback_sidecar,
)
from ..core.bench.profile_summary import (
    ProfileSummaryArtifactCitation,
    build_profile_summary_sidecar,
    profile_summary_artifact_citation_from_path,
)
from ..core.bench.rocm_profiler import Rocprofv3ProfileResult
from ..core.bench.static_kernel_evidence import (
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceSourceReference,
    StaticKernelEvidenceWarning,
    build_static_kernel_evidence_failed,
    build_static_kernel_evidence_unsupported,
    collect_static_kernel_artifacts,
    run_static_kernel_extractors,
)
from ..core.dataset.checksums import sha256_file, stable_json_checksum
from ..core.runtime_evidence import write_json_payload

console = Console(stderr=True)

STATIC_EVIDENCE_NONE = "none"
STATIC_EVIDENCE_AUTO = "auto"


def _profile_output_directory(output_file: Path | None, staging_dir: Path) -> Path:
    """Return the profiler artifact directory for an evaluation run."""

    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.rocprofv3")
    return staging_dir / "rocprofv3"


def _profile_sidecar_path(
    output_file: Path | None,
    profile_result: Rocprofv3ProfileResult,
) -> Path:
    """Return the profile metadata sidecar path for an evaluation run."""

    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.profile.json")
    return profile_result.output_directory / "profile.json"


def _write_profile_sidecar(
    output_file: Path | None,
    profile_result: Rocprofv3ProfileResult | None,
) -> Path | None:
    """Write optional profiler metadata without changing trace JSONL."""

    if profile_result is None:
        return None

    sidecar_path = _profile_sidecar_path(output_file, profile_result)
    try:
        write_json_payload(sidecar_path, profile_result.to_dict())
        console.print(f"[green]Saved profiling metadata to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Profiling metadata skipped: {exc}[/yellow]")
        return None


def _profile_summary_sidecar_path(output_file: Path | None) -> Path | None:
    """Return the optional profile summary sidecar path for a trace output."""

    if output_file is None:
        return None
    return output_file.with_name(f"{output_file.name}.profile-summary.json")


def _write_profile_summary_sidecar(
    output_file: Path | None,
    profile_result: Rocprofv3ProfileResult | None,
    *,
    profile_sidecar_path: Path | None = None,
    run_id: str | None = None,
) -> Path | None:
    """Write optional normalized profile summary without changing trace JSONL."""

    sidecar_path = _profile_summary_sidecar_path(output_file)
    if sidecar_path is None or output_file is None:
        return None
    try:
        trace_sha256 = sha256_file(output_file) if output_file.is_file() else None
        if run_id is None:
            run_id = trace_sha256
        sidecar = build_profile_summary_sidecar(
            profile_result=profile_result,
            trace_path=str(output_file),
            run_id=run_id,
            artifact_citations=_profile_summary_artifact_citations(
                output_file=output_file,
                profile_result=profile_result,
                profile_sidecar_path=profile_sidecar_path,
                trace_sha256=trace_sha256,
            ),
        )
        write_json_payload(sidecar_path, sidecar)
        console.print(f"[green]Saved profile summary to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Profile summary skipped: {exc}[/yellow]")
        return None


def _profile_summary_artifact_citations(
    *,
    output_file: Path,
    profile_result: Rocprofv3ProfileResult | None,
    profile_sidecar_path: Path | None,
    trace_sha256: str | None = None,
) -> list[ProfileSummaryArtifactCitation]:
    """Return compact citations for profile-summary evidence inputs."""

    citations = [
        profile_summary_artifact_citation_from_path(
            kind="trace",
            label="canonical_trace_jsonl",
            path=output_file,
            sha256=trace_sha256,
        )
    ]
    if profile_sidecar_path is not None:
        citations.append(
            profile_summary_artifact_citation_from_path(
                kind="profile_metadata",
                label="rocprofv3_profile_metadata",
                path=profile_sidecar_path,
            )
        )
    if profile_result is not None:
        for artifact in profile_result.artifacts:
            citations.append(
                profile_summary_artifact_citation_from_path(
                    kind="profiler_artifact",
                    label=artifact.kind,
                    path=artifact.path,
                    status=profile_result.status,
                    size_bytes=artifact.size_bytes,
                )
            )
    return citations


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


def _agent_feedback_sidecar_path(output_file: Path | None) -> Path | None:
    """Return the optional agent feedback sidecar path for a trace output."""

    if output_file is None:
        return None
    return output_file.with_name(f"{output_file.name}.agent-feedback.json")


def _write_agent_feedback_sidecar(
    output_file: Path | None,
    traces: list[Trace],
    *,
    solution: Solution | None = None,
    profile_result: Rocprofv3ProfileResult | None,
    static_evidence: StaticKernelEvidenceSidecar | None,
    environment_sidecar_path: Path | None = None,
    profile_sidecar_path: Path | None = None,
    static_evidence_sidecar_path: Path | None = None,
    run_id: str | None = None,
    feedback_target_id: str | None = None,
    feedback_candidate_id: str | None = None,
    feedback_source_sha256: str | None = None,
    feedback_sol_version: str | None = None,
) -> Path | None:
    """Write optional agent feedback metadata without changing trace JSONL."""

    sidecar_path = _agent_feedback_sidecar_path(output_file)
    if sidecar_path is None:
        return None

    try:
        if run_id is None:
            run_id = _agent_feedback_run_id(output_file, traces)
        identity_fields = _agent_feedback_identity_fields(
            output_file,
            traces,
            solution=solution,
            run_id=run_id,
            target_id=feedback_target_id,
            candidate_id=feedback_candidate_id,
            source_sha256=feedback_source_sha256,
        )
        sidecar = build_agent_feedback_sidecar(
            traces=traces,
            profile_result=profile_result,
            static_evidence=static_evidence,
            trace_path=str(output_file) if output_file is not None else None,
            target_id=identity_fields["target_id"],
            run_id=identity_fields["run_id"],
            candidate_id=identity_fields["candidate_id"],
            source_sha256=identity_fields["source_sha256"],
            sol_version=feedback_sol_version,
            artifact_citations=_agent_feedback_artifact_citations(
                output_file=output_file,
                environment_sidecar_path=environment_sidecar_path,
                profile_sidecar_path=profile_sidecar_path,
                static_evidence_sidecar_path=static_evidence_sidecar_path,
            ),
        )
        write_json_payload(sidecar_path, sidecar)
        console.print(f"[green]Saved agent feedback to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Agent feedback metadata skipped: {exc}[/yellow]")
        return None


def _agent_feedback_identity_fields(
    output_file: Path | None,
    traces: Sequence[Trace],
    *,
    solution: Solution | None = None,
    run_id: str | None = None,
    target_id: str | None = None,
    candidate_id: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, str | None]:
    """Derive stable feedback freshness identity from emitted trace data."""

    target_records = [
        {
            "definition": trace.definition,
            "workload_uuid": trace.workload.uuid,
        }
        for trace in traces
    ]
    solution_labels = sorted(
        {trace.solution for trace in traces if trace.solution is not None}
    )

    return {
        "target_id": (
            target_id
            if target_id is not None
            else (stable_json_checksum(target_records) if target_records else None)
        ),
        "run_id": (
            run_id
            if run_id is not None
            else _agent_feedback_run_id(output_file, traces)
        ),
        "candidate_id": (
            candidate_id
            if candidate_id is not None
            else (stable_json_checksum(solution_labels) if solution_labels else None)
        ),
        "source_sha256": (
            source_sha256
            if source_sha256 is not None
            else (solution.hash() if solution is not None else None)
        ),
    }


def _agent_feedback_run_id(
    output_file: Path | None,
    traces: Sequence[Trace],
) -> str | None:
    """Return a stable run id from the persisted trace file or trace payload."""

    if output_file is not None and output_file.is_file():
        return sha256_file(output_file)
    if not traces:
        return None
    return stable_json_checksum([trace.model_dump(mode="json") for trace in traces])


def _agent_feedback_artifact_citations(
    *,
    output_file: Path | None,
    environment_sidecar_path: Path | None,
    profile_sidecar_path: Path | None,
    static_evidence_sidecar_path: Path | None,
    trace_sha256: str | None = None,
) -> list[AgentFeedbackArtifactCitation]:
    """Return compact citations for artifacts written during this CLI run."""

    citations: list[AgentFeedbackArtifactCitation] = []
    if output_file is not None:
        citations.append(
            artifact_citation_from_path(
                kind="trace",
                label="canonical_trace_jsonl",
                path=output_file,
                sha256=trace_sha256,
            )
        )
    for kind, label, path in (
        ("environment", "environment_snapshot", environment_sidecar_path),
        ("profile", "rocprofv3_profile", profile_sidecar_path),
        ("static_evidence", "static_kernel_evidence", static_evidence_sidecar_path),
    ):
        if path is not None:
            citations.append(
                artifact_citation_from_path(kind=kind, label=label, path=path)
            )
    return citations


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
