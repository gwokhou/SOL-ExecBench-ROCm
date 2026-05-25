# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict diagnostic-only static kernel evidence sidecar contract."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field

from sol_execbench.core.data.base_model import BaseModelWithDocstrings


STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION = "sol_execbench.static_kernel_evidence.v1"
_STATIC_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    use_attribute_docstrings=True,
)


class StaticKernelEvidenceStatus(str, Enum):
    """Aggregate and per-artifact status vocabulary."""

    COLLECTED = "collected"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    SKIPPED = "skipped"


class StaticKernelEvidenceReasonCode(str, Enum):
    """Stable reason-code vocabulary for static evidence outcomes."""

    STATIC_EVIDENCE_NOT_REQUESTED = "static_evidence_not_requested"
    STATIC_EVIDENCE_COLLECTED = "static_evidence_collected"
    PARTIAL_ARTIFACT_METADATA = "partial_artifact_metadata"
    PARTIAL_DISASSEMBLY_ONLY = "partial_disassembly_only"
    PARTIAL_METADATA_ONLY = "partial_metadata_only"
    ARTIFACT_UNAVAILABLE = "artifact_unavailable"
    TOOLCHAIN_UNAVAILABLE = "toolchain_unavailable"
    UNSUPPORTED_SOLUTION_TYPE = "unsupported_solution_type"
    UNSUPPORTED_ARCHITECTURE = "unsupported_architecture"
    UNSUPPORTED_ARTIFACT_TYPE = "unsupported_artifact_type"
    EXTRACTOR_FAILED = "extractor_failed"
    EXTRACTOR_TIMEOUT = "extractor_timeout"
    PARSER_FAILED = "parser_failed"


def _validate_status(value: object) -> object:
    if isinstance(value, str):
        return StaticKernelEvidenceStatus(value)
    return value


def _validate_reason_code(value: object) -> object:
    if isinstance(value, str):
        return StaticKernelEvidenceReasonCode(value)
    return value


StaticKernelEvidenceStatusField = Annotated[
    StaticKernelEvidenceStatus,
    BeforeValidator(_validate_status),
]
StaticKernelEvidenceReasonCodeField = Annotated[
    StaticKernelEvidenceReasonCode,
    BeforeValidator(_validate_reason_code),
]


class StaticKernelEvidenceSourceReference(BaseModelWithDocstrings):
    """Source reference used to explain static evidence metadata."""

    model_config = _STATIC_MODEL_CONFIG

    kind: str
    """Reference kind, such as artifact, tool, parser, or note."""
    value: str
    """Reference value or identifier."""
    description: str = ""
    """Human-readable reference description."""


class StaticKernelEvidenceWarning(BaseModelWithDocstrings):
    """Nonfatal warning attached to a static evidence sidecar."""

    model_config = _STATIC_MODEL_CONFIG

    code: str
    """Stable warning code."""
    message: str
    """Human-readable warning message."""
    source_reference: StaticKernelEvidenceSourceReference | None = None
    """Optional reference that explains the warning."""


class StaticKernelEvidenceClassification(BaseModelWithDocstrings):
    """Conservative artifact classification metadata."""

    model_config = _STATIC_MODEL_CONFIG

    metadata_present: bool = False
    """Whether artifact metadata was present."""
    disassembly_present: bool = False
    """Whether disassembly text was present."""
    detected_architectures: list[str] = Field(default_factory=list)
    """Detected gfx architecture identifiers."""
    symbol_count: int | None = Field(default=None, ge=0)
    """Number of detected symbols when known."""


class StaticKernelEvidenceKernel(BaseModelWithDocstrings):
    """Conservative kernel symbol metadata."""

    model_config = _STATIC_MODEL_CONFIG

    name: str
    """Kernel or symbol name."""
    demangled_name: str | None = None
    """Demangled symbol name when known."""
    detected_architectures: list[str] = Field(default_factory=list)
    """Architectures associated with this kernel."""
    source_references: list[StaticKernelEvidenceSourceReference] = Field(
        default_factory=list
    )
    """References supporting this kernel metadata."""


class StaticKernelEvidenceArtifact(BaseModelWithDocstrings):
    """One static evidence artifact entry."""

    model_config = _STATIC_MODEL_CONFIG

    artifact_id: str
    """Stable artifact identifier within the sidecar."""
    artifact_type: str
    """Artifact class such as elf_object, rocm_binary, or disassembly."""
    status: StaticKernelEvidenceStatusField
    """Per-artifact status."""
    reason_code: StaticKernelEvidenceReasonCodeField | None = None
    """Per-artifact reason code."""
    source_path: str | None = None
    """Source artifact path recorded by a future extractor."""
    persisted_path: str | None = None
    """Persisted sidecar-relative artifact path recorded by a future extractor."""
    size_bytes: int | None = Field(default=None, ge=0)
    """Artifact size when known."""
    sha256: str | None = None
    """Artifact digest when known."""
    producer: str | None = None
    """Producer that created or registered this artifact when known."""
    target_architecture: str | None = None
    """Target gfx architecture when known."""
    inspectable: bool = False
    """Whether later static tools may inspect this artifact."""
    classification: StaticKernelEvidenceClassification = Field(
        default_factory=StaticKernelEvidenceClassification
    )
    """Conservative artifact classification."""
    source_references: list[StaticKernelEvidenceSourceReference] = Field(
        default_factory=list
    )
    """References supporting this artifact entry."""


class StaticKernelEvidenceToolRun(BaseModelWithDocstrings):
    """One static evidence tool-run record."""

    model_config = _STATIC_MODEL_CONFIG

    tool_id: str
    """Stable tool identifier."""
    command: list[str] = Field(default_factory=list)
    """Command that a future extractor records."""
    status: StaticKernelEvidenceStatusField
    """Tool-run status."""
    reason_code: StaticKernelEvidenceReasonCodeField | None = None
    """Tool-run reason code."""
    returncode: int | None = None
    """Process return code when a future extractor records one."""
    stdout_tail: str = ""
    """Bounded stdout tail recorded by a future extractor."""
    stderr_tail: str = ""
    """Bounded stderr tail recorded by a future extractor."""
    timeout_seconds: float | None = Field(default=None, ge=0.0)
    """Timeout used by a future extractor when known."""


class StaticKernelEvidenceSidecar(BaseModelWithDocstrings):
    """Strict diagnostic-only static kernel evidence sidecar."""

    model_config = _STATIC_MODEL_CONFIG

    schema_version: Literal[STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION] = (
        STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION
    )
    """Static kernel evidence schema version."""
    status: StaticKernelEvidenceStatusField
    """Aggregate static evidence status."""
    reason_code: StaticKernelEvidenceReasonCodeField
    """Aggregate reason code."""
    diagnostic_only: Literal[True] = True
    """Static evidence is diagnostic metadata only."""
    correctness_authority: Literal[False] = False
    """Static evidence never proves correctness."""
    performance_authority: Literal[False] = False
    """Static evidence never proves performance."""
    timing_authority: Literal[False] = False
    """Static evidence never proves benchmark timing."""
    score_authority: Literal[False] = False
    """Static evidence never proves score validity."""
    paper_parity_authority: Literal[False] = False
    """Static evidence never proves paper parity."""
    leaderboard_authority: Literal[False] = False
    """Static evidence never proves leaderboard readiness."""
    classification: StaticKernelEvidenceClassification = Field(
        default_factory=StaticKernelEvidenceClassification
    )
    """Aggregate conservative classification."""
    artifacts: list[StaticKernelEvidenceArtifact] = Field(default_factory=list)
    """Static evidence artifact entries."""
    tool_runs: list[StaticKernelEvidenceToolRun] = Field(default_factory=list)
    """Static evidence tool-run records."""
    kernels: list[StaticKernelEvidenceKernel] = Field(default_factory=list)
    """Conservative kernel metadata entries."""
    warnings: list[StaticKernelEvidenceWarning] = Field(default_factory=list)
    """Nonfatal static evidence warnings."""
    source_references: list[StaticKernelEvidenceSourceReference] = Field(
        default_factory=list
    )
    """References supporting this sidecar."""

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible sidecar payload."""
        return self.model_dump(mode="json")


def build_static_kernel_evidence_sidecar(
    *,
    status: StaticKernelEvidenceStatus | str,
    reason_code: StaticKernelEvidenceReasonCode | str,
    classification: StaticKernelEvidenceClassification | None = None,
    artifacts: Sequence[StaticKernelEvidenceArtifact] = (),
    tool_runs: Sequence[StaticKernelEvidenceToolRun] = (),
    kernels: Sequence[StaticKernelEvidenceKernel] = (),
    warnings: Sequence[StaticKernelEvidenceWarning] = (),
    source_references: Sequence[StaticKernelEvidenceSourceReference] = (),
) -> StaticKernelEvidenceSidecar:
    """Build a strict static evidence sidecar without collecting artifacts."""

    return StaticKernelEvidenceSidecar(
        status=status,
        reason_code=reason_code,
        classification=classification or StaticKernelEvidenceClassification(),
        artifacts=list(artifacts),
        tool_runs=list(tool_runs),
        kernels=list(kernels),
        warnings=list(warnings),
        source_references=list(source_references),
    )


def build_static_kernel_evidence_skipped(
    reason_code: StaticKernelEvidenceReasonCode | str = (
        StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_NOT_REQUESTED
    ),
) -> StaticKernelEvidenceSidecar:
    """Build a skipped sidecar for unrequested static evidence."""

    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.SKIPPED,
        reason_code=reason_code,
    )


def build_static_kernel_evidence_unavailable(
    reason_code: StaticKernelEvidenceReasonCode | str = (
        StaticKernelEvidenceReasonCode.TOOLCHAIN_UNAVAILABLE
    ),
) -> StaticKernelEvidenceSidecar:
    """Build an unavailable sidecar for missing optional tooling or artifacts."""

    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.UNAVAILABLE,
        reason_code=reason_code,
    )


def build_static_kernel_evidence_unsupported(
    reason_code: StaticKernelEvidenceReasonCode | str = (
        StaticKernelEvidenceReasonCode.UNSUPPORTED_SOLUTION_TYPE
    ),
) -> StaticKernelEvidenceSidecar:
    """Build an unsupported sidecar for unsupported solution or artifact classes."""

    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.UNSUPPORTED,
        reason_code=reason_code,
    )


def build_static_kernel_evidence_failed(
    reason_code: StaticKernelEvidenceReasonCode | str = (
        StaticKernelEvidenceReasonCode.EXTRACTOR_FAILED
    ),
) -> StaticKernelEvidenceSidecar:
    """Build a failed sidecar for failed optional evidence extraction."""

    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.FAILED,
        reason_code=reason_code,
    )


def build_static_kernel_evidence_partial(
    reason_code: StaticKernelEvidenceReasonCode | str = (
        StaticKernelEvidenceReasonCode.PARTIAL_ARTIFACT_METADATA
    ),
) -> StaticKernelEvidenceSidecar:
    """Build a partial sidecar for incomplete optional static evidence."""

    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.PARTIAL,
        reason_code=reason_code,
    )


_PRIMARY_ARTIFACT_NAME = "benchmark_kernel.so"
_COMPILER_OUTPUT_SUFFIXES = {".log", ".txt"}
_STATIC_ARTIFACT_SUFFIXES = {
    ".hsaco": "hsaco",
    ".co": "code_object",
    ".o": "object_file",
}


def collect_static_kernel_artifacts(
    *,
    build_directory: Path,
    evidence_directory: Path,
    primary_artifact_name: str = _PRIMARY_ARTIFACT_NAME,
    sidecar_base_directory: Path | None = None,
    producer: str = "hip_cpp_build",
    target_architecture: str | None = None,
) -> StaticKernelEvidenceSidecar:
    """Persist current-build static artifacts into an evidence directory."""

    build_root = build_directory.resolve()
    if not build_root.is_dir():
        return build_static_kernel_evidence_unavailable(
            StaticKernelEvidenceReasonCode.ARTIFACT_UNAVAILABLE
        )

    primary_artifact = build_root / primary_artifact_name
    if not _is_contained_file(primary_artifact, build_root):
        return build_static_kernel_evidence_unavailable(
            StaticKernelEvidenceReasonCode.ARTIFACT_UNAVAILABLE
        )

    evidence_root = evidence_directory.resolve()
    sidecar_base = (
        sidecar_base_directory.resolve()
        if sidecar_base_directory is not None
        else evidence_root
    )
    artifact_root = evidence_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    artifacts = [
        _persist_static_artifact(
            source_path=path,
            build_root=build_root,
            artifact_root=artifact_root,
            sidecar_base=sidecar_base,
            producer=producer,
            target_architecture=target_architecture,
        )
        for path in _discover_static_artifact_paths(
            build_root=build_root,
            evidence_root=evidence_root,
            primary_artifact_name=primary_artifact_name,
        )
    ]

    if not artifacts:
        return build_static_kernel_evidence_unavailable(
            StaticKernelEvidenceReasonCode.ARTIFACT_UNAVAILABLE
        )

    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        artifacts=artifacts,
        classification=StaticKernelEvidenceClassification(
            metadata_present=True,
            detected_architectures=(
                [target_architecture] if target_architecture is not None else []
            ),
        ),
    )


def _discover_static_artifact_paths(
    *,
    build_root: Path,
    evidence_root: Path,
    primary_artifact_name: str,
) -> tuple[Path, ...]:
    primary_artifact = build_root / primary_artifact_name
    candidates: list[Path] = []
    if _is_contained_file(primary_artifact, build_root):
        candidates.append(primary_artifact.resolve())

    for path in sorted(build_root.rglob("*")):
        if not _is_contained_file(path, build_root):
            continue
        resolved = path.resolve()
        if _is_relative_to(resolved, evidence_root):
            continue
        if resolved == primary_artifact.resolve():
            continue
        if _static_artifact_type(path) is None:
            continue
        candidates.append(resolved)

    return tuple(dict.fromkeys(candidates))


def _persist_static_artifact(
    *,
    source_path: Path,
    build_root: Path,
    artifact_root: Path,
    sidecar_base: Path,
    producer: str,
    target_architecture: str | None,
) -> StaticKernelEvidenceArtifact:
    source_relative_path = source_path.resolve().relative_to(build_root)
    persisted_path = artifact_root / source_relative_path
    persisted_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, persisted_path)

    return StaticKernelEvidenceArtifact(
        artifact_id=_artifact_id(source_relative_path),
        artifact_type=_static_artifact_type(source_path) or "unknown",
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        source_path=source_relative_path.as_posix(),
        persisted_path=_relative_path_string(persisted_path, sidecar_base),
        size_bytes=persisted_path.stat().st_size,
        sha256=_sha256_file(persisted_path),
        producer=producer,
        target_architecture=target_architecture,
        inspectable=_is_inspectable_artifact(source_path),
        classification=StaticKernelEvidenceClassification(
            metadata_present=True,
            detected_architectures=(
                [target_architecture] if target_architecture is not None else []
            ),
        ),
        source_references=[
            StaticKernelEvidenceSourceReference(
                kind="producer",
                value=producer,
                description="Build step that registered this current-build artifact.",
            )
        ],
    )


def _static_artifact_type(path: Path) -> str | None:
    name = path.name
    suffix = path.suffix
    if name == _PRIMARY_ARTIFACT_NAME or (
        name.startswith("benchmark_kernel") and suffix == ".so"
    ):
        return "shared_library"
    if suffix in _STATIC_ARTIFACT_SUFFIXES:
        return _STATIC_ARTIFACT_SUFFIXES[suffix]
    if suffix in _COMPILER_OUTPUT_SUFFIXES and _is_compiler_output_name(name):
        return "compiler_output"
    return None


def _is_compiler_output_name(name: str) -> bool:
    normalized = name.lower()
    return any(marker in normalized for marker in ("build", "compile", "compiler"))


def _is_contained_file(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        return False
    return resolved.is_file() and _is_relative_to(resolved, root)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _artifact_id(relative_path: Path) -> str:
    normalized = relative_path.as_posix().replace("/", "-")
    return f"artifact-{normalized}"


def _relative_path_string(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_inspectable_artifact(path: Path) -> bool:
    return path.suffix in {".so", ".hsaco", ".co", ".o"}
