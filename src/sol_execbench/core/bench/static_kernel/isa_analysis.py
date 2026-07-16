# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Optional machine-readable AMD ISA analysis for static evidence."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path
import re

from sol_execbench.core.bench.static_kernel.amdgpu_metadata import (
    extract_amdgpu_targets,
)
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticIsaAnalysis,
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceClassification,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceStatus,
    StaticKernelEvidenceToolRun,
)
from sol_execbench.core.bench.static_kernel.extractor_routing import (
    artifact_persisted_path,
)
from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.platform.amdgpu_code_object import (
    bundled_architectures,
    extract_code_object,
)
from sol_execbench.core.platform.isa_validation import analyze_isa_disassembly


_ARCHITECTURE = re.compile(r"gfx[0-9a-z]+")
_MAX_METADATA_SCAN_BYTES = 128 * 1024 * 1024


def collect_static_isa_analyses(
    *,
    artifacts: Sequence[StaticKernelEvidenceArtifact],
    evidence_root: Path,
    sidecar_base: Path,
    timeout_seconds: float,
) -> tuple[
    list[StaticIsaAnalysis],
    list[StaticKernelEvidenceToolRun],
    list[StaticKernelEvidenceArtifact],
]:
    """Decode all discoverable AMDGPU targets without affecting benchmark authority."""

    analyses: list[StaticIsaAnalysis] = []
    tool_runs: list[StaticKernelEvidenceToolRun] = []
    generated: list[StaticKernelEvidenceArtifact] = []
    seen: set[tuple[str, str]] = set()
    for artifact in artifacts:
        path = artifact_persisted_path(artifact, sidecar_base)
        if path is None or not path.is_file() or not artifact.inspectable:
            continue
        try:
            architectures = _architectures(
                artifact, path, evidence_root, timeout_seconds
            )
        except (OSError, RuntimeError, ValueError) as exc:
            analyses.append(_failed_analysis(artifact.artifact_id, exc))
            tool_runs.append(_failed_tool_run(exc, timeout_seconds))
            continue
        for architecture in architectures:
            try:
                collected = _collect_architecture(
                    artifact=artifact,
                    binary=path,
                    architecture=architecture,
                    evidence_root=evidence_root,
                    sidecar_base=sidecar_base,
                    timeout_seconds=timeout_seconds,
                    seen=seen,
                )
                if collected is None:
                    continue
                analysis, tool_run, new_artifacts = collected
                analyses.append(analysis)
                tool_runs.append(tool_run)
                generated.extend(new_artifacts)
            except Exception as exc:
                analyses.append(
                    _failed_analysis(artifact.artifact_id, exc, architecture)
                )
                tool_runs.append(_failed_tool_run(exc, timeout_seconds, architecture))
    return analyses, tool_runs, generated


def _collect_architecture(
    *,
    artifact: StaticKernelEvidenceArtifact,
    binary: Path,
    architecture: str,
    evidence_root: Path,
    sidecar_base: Path,
    timeout_seconds: float,
    seen: set[tuple[str, str]],
) -> (
    tuple[
        StaticIsaAnalysis,
        StaticKernelEvidenceToolRun,
        list[StaticKernelEvidenceArtifact],
    ]
    | None
):
    architecture = _normalize_architecture(architecture)
    workspace = _artifact_workspace(evidence_root, artifact.artifact_id) / architecture
    extracted = extract_code_object(
        binary, architecture, workspace, timeout_seconds=timeout_seconds
    )
    identity = (extracted.sha256, architecture)
    if identity in seen:
        return None
    seen.add(identity)
    disassembly_path = workspace / f"{architecture}.isa.txt"
    disassembly_path.parent.mkdir(parents=True, exist_ok=True)
    disassembly_path.write_text(extracted.disassembly, encoding="utf-8")
    decoded = analyze_isa_disassembly(
        architecture, extracted.disassembly, allow_download=True
    )
    analysis = StaticIsaAnalysis(
        artifact_id=artifact.artifact_id,
        architecture=architecture,
        status=StaticKernelEvidenceStatus.COLLECTED,
        decoded_instruction_count=decoded.decoded_instruction_count,
        functional_group_counts=dict(decoded.functional_group_counts),
        functional_subgroup_counts=dict(decoded.functional_subgroup_counts),
        observed_matrix_units=list(decoded.observed_matrix_units),
        code_object_sha256=extracted.sha256,
        disassembly_sha256=extracted.disassembly_sha256,
        spec_provenance=decoded.provenance.to_dict(),
    )
    tool_run = StaticKernelEvidenceToolRun(
        tool_id="amd-isa",
        command=["amd-isa", architecture],
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        stdout_tail=f"decoded {decoded.decoded_instruction_count} instructions",
        timeout_seconds=timeout_seconds,
    )
    return (
        analysis,
        tool_run,
        _generated_artifacts(
            artifact.artifact_id,
            architecture,
            extracted.path,
            disassembly_path,
            sidecar_base,
        ),
    )


def _architectures(
    artifact: StaticKernelEvidenceArtifact,
    path: Path,
    evidence_root: Path,
    timeout_seconds: float,
) -> tuple[str, ...]:
    if artifact.target_architecture:
        return (_normalize_architecture(artifact.target_architecture),)
    metadata_targets = (
        extract_amdgpu_targets(path.read_bytes())
        if path.stat().st_size <= _MAX_METADATA_SCAN_BYTES
        else ()
    )
    if metadata_targets:
        return metadata_targets
    return bundled_architectures(
        path,
        _artifact_workspace(evidence_root, artifact.artifact_id),
        timeout_seconds=timeout_seconds,
    )


def _artifact_workspace(evidence_root: Path, artifact_id: str) -> Path:
    digest = hashlib.sha256(artifact_id.encode()).hexdigest()[:16]
    return evidence_root / "isa" / digest


def _normalize_architecture(value: str) -> str:
    architecture = value.lower().split(":", maxsplit=1)[0].strip()
    if _ARCHITECTURE.fullmatch(architecture) is None:
        raise ValueError(f"invalid AMDGPU architecture: {architecture}")
    return architecture


def _generated_artifacts(
    artifact_id: str,
    architecture: str,
    code_object: Path,
    disassembly: Path,
    sidecar_base: Path,
) -> list[StaticKernelEvidenceArtifact]:
    values: list[StaticKernelEvidenceArtifact] = []
    for suffix, artifact_type, path in (
        ("code-object", "code_object", code_object),
        ("disassembly", "isa_disassembly", disassembly),
    ):
        values.append(
            StaticKernelEvidenceArtifact(
                artifact_id=f"{artifact_id}-{architecture}-isa-{suffix}",
                artifact_type=artifact_type,
                status=StaticKernelEvidenceStatus.COLLECTED,
                reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
                persisted_path=_relative(path, sidecar_base),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                producer="amd-isa",
                target_architecture=architecture,
                inspectable=artifact_type == "code_object",
                classification=StaticKernelEvidenceClassification(
                    metadata_present=True,
                    disassembly_present=artifact_type == "isa_disassembly",
                    detected_architectures=[architecture],
                ),
            )
        )
    return values


def _failed_analysis(
    artifact_id: str, exc: Exception, architecture: str = "unknown"
) -> StaticIsaAnalysis:
    return StaticIsaAnalysis(
        artifact_id=artifact_id,
        architecture=architecture,
        status=StaticKernelEvidenceStatus.UNAVAILABLE,
        reason_code=_reason_code(exc),
    )


def _failed_tool_run(
    exc: Exception, timeout_seconds: float, architecture: str = "unknown"
) -> StaticKernelEvidenceToolRun:
    return StaticKernelEvidenceToolRun(
        tool_id="amd-isa",
        command=["amd-isa", architecture],
        status=StaticKernelEvidenceStatus.UNAVAILABLE,
        reason_code=StaticKernelEvidenceReasonCode.TOOLCHAIN_UNAVAILABLE,
        stderr_tail=str(exc)[-4000:],
        timeout_seconds=timeout_seconds,
    )


def _reason_code(exc: Exception) -> str:
    name = type(exc).__name__
    return {
        "IsaSpecUnavailableError": "isa_spec_unavailable",
        "IsaDownloadError": "isa_download_failed",
        "IsaIntegrityError": "isa_integrity_failed",
        "IsaHelperBuildError": "isa_helper_build_failed",
        "IsaDecodeError": "isa_decode_failed",
        "IsaProtocolError": "isa_protocol_failed",
        "FileNotFoundError": "isa_artifact_tool_unavailable",
    }.get(name, "isa_artifact_extraction_failed")


def _relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return path.resolve().as_posix()


__all__ = ["collect_static_isa_analyses"]
