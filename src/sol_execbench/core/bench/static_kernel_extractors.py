# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Static kernel evidence extractor routing and execution."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from sol_execbench.core.bench.static_kernel_artifacts import (
    _relative_path_string,
    _sha256_file,
)
from sol_execbench.core.bench.static_kernel_evidence_builders import (
    build_static_kernel_evidence_sidecar,
)
from sol_execbench.core.bench.static_kernel_evidence_models import (
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceClassification,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceSourceReference,
    StaticKernelEvidenceStatus,
    StaticKernelEvidenceToolRun,
    StaticKernelEvidenceWarning,
)
from sol_execbench.core.bench.static_kernel_status import (
    aggregate_extractor_reason_value,
    aggregate_extractor_status_value,
)
from sol_execbench.core.environment import ProbeCompletedProcess
from sol_execbench.core.toolchain import (
    ProbeRunner,
    ToolchainArtifactType,
    ToolchainCapability,
    ToolchainEvidenceLevel,
    ToolchainRoutingDecision,
    ToolchainRoutingRequest,
    ToolchainStatus,
    Which,
    build_toolchain_routing_report,
    default_toolchain_registry,
)


ExtractorRunner = Callable[[list[str], float], ProbeCompletedProcess]

_STATIC_EXTRACTOR_TOOL_IDS = ("llvm-objdump", "readelf")
_RAW_OUTPUT_LIMIT = 64 * 1024
_TAIL_LIMIT = 4000


def run_static_kernel_extractors(
    *,
    artifacts: Sequence[StaticKernelEvidenceArtifact],
    evidence_directory: Path,
    sidecar_base_directory: Path | None = None,
    timeout_seconds: float = 10.0,
    runner: ExtractorRunner | None = None,
    probe_runner: ProbeRunner | None = None,
    which: Which = shutil.which,
    registry: Sequence[ToolchainCapability] | None = None,
) -> StaticKernelEvidenceSidecar:
    """Run routed bounded static extractors for persisted artifacts."""

    evidence_root = evidence_directory.resolve()
    sidecar_base = (
        sidecar_base_directory.resolve()
        if sidecar_base_directory is not None
        else evidence_root
    )
    effective_registry = list(registry) if registry is not None else None
    tool_runs: list[StaticKernelEvidenceToolRun] = []
    warnings: list[StaticKernelEvidenceWarning] = []
    output_artifacts: list[StaticKernelEvidenceArtifact] = []

    for artifact in artifacts:
        artifact_type = _toolchain_artifact_type_for_static_artifact(artifact)
        if artifact_type is None:
            tool_runs.append(
                StaticKernelEvidenceToolRun(
                    tool_id="static-extractor",
                    command=[],
                    status=StaticKernelEvidenceStatus.UNSUPPORTED,
                    reason_code=(
                        StaticKernelEvidenceReasonCode.UNSUPPORTED_ARTIFACT_TYPE
                    ),
                    stderr_tail=(
                        f"{artifact.artifact_type} is not supported by "
                        "llvm-objdump/readelf extraction."
                    ),
                    timeout_seconds=timeout_seconds,
                )
            )
            continue

        artifact_path = _artifact_persisted_path(artifact, sidecar_base)
        if artifact_path is None or not artifact_path.is_file():
            tool_runs.append(
                StaticKernelEvidenceToolRun(
                    tool_id="static-extractor",
                    command=[],
                    status=StaticKernelEvidenceStatus.UNAVAILABLE,
                    reason_code=StaticKernelEvidenceReasonCode.ARTIFACT_UNAVAILABLE,
                    stderr_tail=f"{artifact.artifact_id} persisted artifact is missing.",
                    timeout_seconds=timeout_seconds,
                )
            )
            continue

        for tool_id in _STATIC_EXTRACTOR_TOOL_IDS:
            route_decision = _route_static_tool(
                tool_id=tool_id,
                artifact_type=artifact_type,
                registry=effective_registry,
                runner=probe_runner,
                which=which,
                timeout_seconds=timeout_seconds,
            )
            if route_decision is None:
                tool_runs.append(
                    _tool_run_from_route_decision(
                        tool_id=tool_id,
                        command=[],
                        decision_status=ToolchainStatus.UNAVAILABLE,
                        reason_code=StaticKernelEvidenceReasonCode.TOOLCHAIN_UNAVAILABLE,
                        reason="No route decision was produced.",
                        timeout_seconds=timeout_seconds,
                    )
                )
                continue
            command = _extractor_command(tool_id, artifact_path)
            if route_decision.status != ToolchainStatus.AVAILABLE:
                tool_runs.append(
                    _tool_run_from_route_decision(
                        tool_id=tool_id,
                        command=command,
                        decision_status=route_decision.status,
                        reason_code=_reason_for_route_status(route_decision.status),
                        reason=route_decision.reason,
                        timeout_seconds=timeout_seconds,
                    )
                )
                warnings.append(
                    StaticKernelEvidenceWarning(
                        code=f"{tool_id}_not_executed",
                        message=route_decision.reason,
                        source_reference=StaticKernelEvidenceSourceReference(
                            kind="toolchain_route",
                            value=tool_id,
                            description=route_decision.reason_code,
                        ),
                    )
                )
                continue

            tool_run, raw_artifact = _run_static_extractor(
                tool_id=tool_id,
                command=command,
                artifact=artifact,
                evidence_root=evidence_root,
                sidecar_base=sidecar_base,
                timeout_seconds=timeout_seconds,
                runner=runner,
            )
            tool_runs.append(tool_run)
            if raw_artifact is not None:
                output_artifacts.append(raw_artifact)

    all_artifacts = list(artifacts) + output_artifacts
    return build_static_kernel_evidence_sidecar(
        status=_aggregate_extractor_status(tool_runs),
        reason_code=_aggregate_extractor_reason(tool_runs),
        artifacts=all_artifacts,
        tool_runs=tool_runs,
        warnings=warnings,
        classification=_classification_from_tool_runs(tool_runs, artifacts),
    )


def _route_static_tool(
    *,
    tool_id: str,
    artifact_type: ToolchainArtifactType,
    registry: Sequence[ToolchainCapability] | None,
    runner: ProbeRunner | None,
    which: Which,
    timeout_seconds: float,
) -> ToolchainRoutingDecision | None:
    effective_registry = (
        list(registry) if registry is not None else default_toolchain_registry()
    )
    tool_registry = [entry for entry in effective_registry if entry.tool_id == tool_id]
    report = build_toolchain_routing_report(
        ToolchainRoutingRequest(
            evidence_level=ToolchainEvidenceLevel.STATIC,
            artifact_type=artifact_type,
        ),
        registry=tool_registry,
        runner=runner,
        which=which,
        timeout_seconds=timeout_seconds,
    )
    for decision in report.decisions:
        if decision.tool_id == tool_id:
            return decision
    return None


def _run_static_extractor(
    *,
    tool_id: str,
    command: list[str],
    artifact: StaticKernelEvidenceArtifact,
    evidence_root: Path,
    sidecar_base: Path,
    timeout_seconds: float,
    runner: ExtractorRunner | None,
) -> tuple[StaticKernelEvidenceToolRun, StaticKernelEvidenceArtifact | None]:
    effective_runner = runner or _run_extractor_command
    try:
        completed = effective_runner(command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raw_artifact = _write_raw_extractor_output(
            tool_id=tool_id,
            artifact_id=artifact.artifact_id,
            evidence_root=evidence_root,
            sidecar_base=sidecar_base,
            stdout=_decode_output(exc.stdout),
            stderr=_decode_output(exc.stderr),
        )
        return (
            StaticKernelEvidenceToolRun(
                tool_id=tool_id,
                command=command,
                status=StaticKernelEvidenceStatus.FAILED,
                reason_code=StaticKernelEvidenceReasonCode.EXTRACTOR_TIMEOUT,
                stdout_tail=_tail_text(exc.stdout),
                stderr_tail=_tail_text(exc.stderr),
                timeout_seconds=timeout_seconds,
                raw_output_path=raw_artifact.persisted_path,
            ),
            raw_artifact,
        )
    except OSError as exc:
        raw_artifact = _write_raw_extractor_output(
            tool_id=tool_id,
            artifact_id=artifact.artifact_id,
            evidence_root=evidence_root,
            sidecar_base=sidecar_base,
            stdout="",
            stderr=str(exc),
        )
        return (
            StaticKernelEvidenceToolRun(
                tool_id=tool_id,
                command=command,
                status=StaticKernelEvidenceStatus.FAILED,
                reason_code=StaticKernelEvidenceReasonCode.EXTRACTOR_FAILED,
                stderr_tail=_tail_text(str(exc)),
                timeout_seconds=timeout_seconds,
                raw_output_path=raw_artifact.persisted_path,
            ),
            raw_artifact,
        )

    raw_artifact = _write_raw_extractor_output(
        tool_id=tool_id,
        artifact_id=artifact.artifact_id,
        evidence_root=evidence_root,
        sidecar_base=sidecar_base,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    status = (
        StaticKernelEvidenceStatus.COLLECTED
        if completed.returncode == 0
        else StaticKernelEvidenceStatus.FAILED
    )
    reason_code = (
        StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED
        if completed.returncode == 0
        else StaticKernelEvidenceReasonCode.EXTRACTOR_FAILED
    )
    return (
        StaticKernelEvidenceToolRun(
            tool_id=tool_id,
            command=command,
            status=status,
            reason_code=reason_code,
            returncode=completed.returncode,
            stdout_tail=_tail_text(completed.stdout),
            stderr_tail=_tail_text(completed.stderr),
            timeout_seconds=timeout_seconds,
            raw_output_path=raw_artifact.persisted_path,
        ),
        raw_artifact,
    )


def _run_extractor_command(
    command: list[str], timeout_seconds: float
) -> ProbeCompletedProcess:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return ProbeCompletedProcess(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _write_raw_extractor_output(
    *,
    tool_id: str,
    artifact_id: str,
    evidence_root: Path,
    sidecar_base: Path,
    stdout: object,
    stderr: object,
) -> StaticKernelEvidenceArtifact:
    output_path = evidence_root / "extractors" / artifact_id / f"{tool_id}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = _bounded_raw_output(stdout=stdout, stderr=stderr)
    output_path.write_text(output_text)
    return StaticKernelEvidenceArtifact(
        artifact_id=f"{artifact_id}-{tool_id}-raw-output",
        artifact_type="extractor_raw_output",
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        persisted_path=_relative_path_string(output_path, sidecar_base),
        size_bytes=output_path.stat().st_size,
        sha256=_sha256_file(output_path),
        producer=tool_id,
        inspectable=False,
        classification=StaticKernelEvidenceClassification(metadata_present=True),
    )


def _tool_run_from_route_decision(
    *,
    tool_id: str,
    command: list[str],
    decision_status: ToolchainStatus,
    reason_code: StaticKernelEvidenceReasonCode,
    reason: str,
    timeout_seconds: float,
) -> StaticKernelEvidenceToolRun:
    return StaticKernelEvidenceToolRun(
        tool_id=tool_id,
        command=command,
        status=_status_for_route_status(decision_status),
        reason_code=reason_code,
        stderr_tail=reason,
        timeout_seconds=timeout_seconds,
    )


def _extractor_command(tool_id: str, artifact_path: Path) -> list[str]:
    if tool_id == "llvm-objdump":
        return [tool_id, "--disassemble", str(artifact_path)]
    if tool_id == "readelf":
        return [tool_id, "--headers", "--wide", str(artifact_path)]
    raise ValueError(f"unsupported static extractor: {tool_id}")


def _toolchain_artifact_type_for_static_artifact(
    artifact: StaticKernelEvidenceArtifact,
) -> ToolchainArtifactType | None:
    if artifact.artifact_type in {"shared_library", "hsaco", "code_object"}:
        return ToolchainArtifactType.ROCM_BINARY
    if artifact.artifact_type == "object_file":
        return ToolchainArtifactType.ELF_OBJECT
    return None


def _artifact_persisted_path(
    artifact: StaticKernelEvidenceArtifact,
    sidecar_base: Path,
) -> Path | None:
    if artifact.persisted_path is None:
        return None
    path = Path(artifact.persisted_path)
    if path.is_absolute():
        return path
    return sidecar_base / path


def _status_for_route_status(status: ToolchainStatus) -> StaticKernelEvidenceStatus:
    if status == ToolchainStatus.UNSUPPORTED_ARTIFACT:
        return StaticKernelEvidenceStatus.UNSUPPORTED
    if status == ToolchainStatus.UNSUPPORTED_ARCH:
        return StaticKernelEvidenceStatus.UNSUPPORTED
    if status == ToolchainStatus.FAILED:
        return StaticKernelEvidenceStatus.FAILED
    return StaticKernelEvidenceStatus.UNAVAILABLE


def _reason_for_route_status(
    status: ToolchainStatus,
) -> StaticKernelEvidenceReasonCode:
    if status == ToolchainStatus.UNSUPPORTED_ARTIFACT:
        return StaticKernelEvidenceReasonCode.UNSUPPORTED_ARTIFACT_TYPE
    if status == ToolchainStatus.UNSUPPORTED_ARCH:
        return StaticKernelEvidenceReasonCode.UNSUPPORTED_ARCHITECTURE
    if status == ToolchainStatus.FAILED:
        return StaticKernelEvidenceReasonCode.EXTRACTOR_FAILED
    return StaticKernelEvidenceReasonCode.TOOLCHAIN_UNAVAILABLE


def _aggregate_extractor_status(
    tool_runs: Sequence[StaticKernelEvidenceToolRun],
) -> StaticKernelEvidenceStatus:
    return aggregate_extractor_status_value(
        tuple(tool_runs),
        collected=StaticKernelEvidenceStatus.COLLECTED,
        partial=StaticKernelEvidenceStatus.PARTIAL,
        failed=StaticKernelEvidenceStatus.FAILED,
        unavailable=StaticKernelEvidenceStatus.UNAVAILABLE,
    )


def _aggregate_extractor_reason(
    tool_runs: Sequence[StaticKernelEvidenceToolRun],
) -> StaticKernelEvidenceReasonCode:
    status = _aggregate_extractor_status(tool_runs)
    return aggregate_extractor_reason_value(
        tuple(tool_runs),
        status=status,
        collected_status=StaticKernelEvidenceStatus.COLLECTED,
        partial_status=StaticKernelEvidenceStatus.PARTIAL,
        failed_status=StaticKernelEvidenceStatus.FAILED,
        collected_reason=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        partial_reason=StaticKernelEvidenceReasonCode.PARTIAL_ARTIFACT_METADATA,
        partial_disassembly_reason=StaticKernelEvidenceReasonCode.PARTIAL_DISASSEMBLY_ONLY,
        failed_reason=StaticKernelEvidenceReasonCode.EXTRACTOR_FAILED,
        timeout_reason=StaticKernelEvidenceReasonCode.EXTRACTOR_TIMEOUT,
        unavailable_reason=StaticKernelEvidenceReasonCode.TOOLCHAIN_UNAVAILABLE,
    )


def _classification_from_tool_runs(
    tool_runs: Sequence[StaticKernelEvidenceToolRun],
    artifacts: Sequence[StaticKernelEvidenceArtifact],
) -> StaticKernelEvidenceClassification:
    metadata_present = any(
        run.tool_id == "readelf" and run.status == StaticKernelEvidenceStatus.COLLECTED
        for run in tool_runs
    )
    disassembly_present = any(
        run.tool_id == "llvm-objdump"
        and run.status == StaticKernelEvidenceStatus.COLLECTED
        and bool(run.stdout_tail.strip())
        for run in tool_runs
    )
    detected_architectures = sorted(
        {
            architecture
            for artifact in artifacts
            for architecture in artifact.classification.detected_architectures
        }
    )
    return StaticKernelEvidenceClassification(
        metadata_present=metadata_present,
        disassembly_present=disassembly_present,
        detected_architectures=detected_architectures,
    )


def _bounded_raw_output(*, stdout: object, stderr: object) -> str:
    text = f"stdout:\n{_decode_output(stdout)}\n\nstderr:\n{_decode_output(stderr)}"
    return text[-_RAW_OUTPUT_LIMIT:]


def _decode_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _tail_text(value: object, limit: int = _TAIL_LIMIT) -> str:
    return _decode_output(value)[-limit:]
