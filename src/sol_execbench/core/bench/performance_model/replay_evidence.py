# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Trusted-input and cross-pass evidence for diagnostic counter replay."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.diagnostic_sidecar import (
    CurrentDiagnosticSidecarAuthority,
    DiagnosticSidecarStatus,
)
from sol_execbench.core.bench.performance_model.schema_versions import (
    PerformanceArtifactSchema,
    PerformanceEvidenceComponentKind,
)
from sol_execbench.core.data.base_model import StrictArtifactModel
from sol_execbench.core.data.json_utils import load_json_file, load_jsonl_file
from sol_execbench.core.data.trace import CacheClearEvidence
from sol_execbench.core.evidence.runtime_evidence.models import (
    RuntimeGPUTelemetry,
)
from sol_execbench.core.integrity import (
    SHA256Digest,
    sha256_file,
    stable_json_checksum,
)
from sol_execbench.core.platform.hardware import PCIeTopologyIdentity

REPLAY_PROTOCOL_VERSION = "gfx1200_counter_replay.v1"
REPLAY_WARMUP_RUNS = 10
REPLAY_EVIDENCE_ITERATIONS = 5
MAX_REPLAY_PASSES = 4
MAX_DISPATCHES_PER_MARKER = 256
MAX_REPLAY_BUNDLE_BYTES = 512 * 1024 * 1024
RAW_REPLAY_GLOB = "performance-replay-raw-*.jsonl"

_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

if TYPE_CHECKING:
    from sol_execbench.core.bench.performance_model.timing_evidence import (
        PerformanceTimingEvidenceSidecar,
    )


class ReplayProtocol(StrictArtifactModel):
    """Frozen diagnostic replay protocol independent of benchmark config."""

    model_config = _CONFIG

    version: Literal["gfx1200_counter_replay.v1"] = REPLAY_PROTOCOL_VERSION
    warmup_runs: Literal[10] = REPLAY_WARMUP_RUNS
    evidence_iterations: Literal[5] = REPLAY_EVIDENCE_ITERATIONS
    max_passes: Literal[4] = MAX_REPLAY_PASSES
    max_dispatches_per_marker: Literal[256] = MAX_DISPATCHES_PER_MARKER
    max_counter_csv_bytes: Literal[134217728] = 128 * 1024 * 1024
    max_bundle_bytes: Literal[536870912] = MAX_REPLAY_BUNDLE_BYTES


class RawPerformanceReplayRecord(StrictArtifactModel):
    """One trusted-driver replay process record before trace publication."""

    model_config = _CONFIG

    pid: int = Field(gt=0)
    parent_pid: int = Field(gt=0)
    process_executable_sha256: SHA256Digest
    pass_index: int = Field(ge=1, le=MAX_REPLAY_PASSES)
    workload_uuid: str = Field(min_length=1)
    input_sha256: SHA256Digest
    cache_identity_sha256: SHA256Digest
    marker_ranges: list[str] = Field(
        min_length=REPLAY_EVIDENCE_ITERATIONS,
        max_length=REPLAY_EVIDENCE_ITERATIONS,
    )
    protocol: ReplayProtocol = Field(default_factory=ReplayProtocol)


class ReplayProcessEvidence(StrictArtifactModel):
    """Content-bound replay process admitted into the final sidecar."""

    model_config = _CONFIG

    pass_index: int = Field(ge=1, le=MAX_REPLAY_PASSES)
    pid: int = Field(gt=0)
    parent_pid: int = Field(gt=0)
    process_executable_sha256: SHA256Digest
    workload_uuid: str
    input_sha256: SHA256Digest
    cache_identity_sha256: SHA256Digest
    marker_ranges: list[str]
    dispatch_sequence_digest: SHA256Digest
    process_fingerprint: SHA256Digest


class PerformanceReplayEvidenceSidecar(CurrentDiagnosticSidecarAuthority):
    """Fail-closed replay identity and alignment evidence."""

    model_config = _CONFIG
    current_schema_version = (
        PerformanceArtifactSchema.PERFORMANCE_EVIDENCE_COMPONENT
    )
    current_artifact_kind = PerformanceEvidenceComponentKind.REPLAY

    schema_version: Literal[
        PerformanceArtifactSchema.PERFORMANCE_EVIDENCE_COMPONENT
    ] = PerformanceArtifactSchema.PERFORMANCE_EVIDENCE_COMPONENT
    artifact_kind: Literal[PerformanceEvidenceComponentKind.REPLAY] = (
        PerformanceEvidenceComponentKind.REPLAY
    )
    status: DiagnosticSidecarStatus
    run_id: SHA256Digest
    candidate_sha256: SHA256Digest
    canonical_input_sha256: SHA256Digest
    protocol: ReplayProtocol = Field(default_factory=ReplayProtocol)
    processes: list[ReplayProcessEvidence] = Field(default_factory=list)
    environment: list[RuntimeGPUTelemetry] = Field(default_factory=list)
    artifact_sha256: dict[str, SHA256Digest] = Field(default_factory=dict)
    alignment_digest: SHA256Digest | None = None
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_matches_reasons(self) -> PerformanceReplayEvidenceSidecar:
        """Reject contradictory availability and replay identity."""
        pass_indices = [process.pass_index for process in self.processes]
        if len(pass_indices) != len(set(pass_indices)):
            raise ValueError("replay evidence repeats pass index")
        if self.status is DiagnosticSidecarStatus.AVAILABLE and (
            self.reason_codes or self.alignment_digest is None
        ):
            raise ValueError("available replay requires aligned evidence")
        return self

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible replay evidence."""
        return self.model_dump(mode="json")


def build_performance_replay_evidence(
    *,
    staging_dir: Path,
    run_id: str,
    candidate_sha256: str,
    canonical_timing_path: Path,
    artifact_paths: list[Path],
    counter_paths: list[Path] | None = None,
    expected_gpu_id: str | None = None,
    expected_gpu_bdf: str | None = None,
    expected_pcie_topology: PCIeTopologyIdentity | None = None,
    environment: list[RuntimeGPUTelemetry] | None = None,
) -> PerformanceReplayEvidenceSidecar:
    """Build replay evidence and reject identity drift without guessing."""
    timing = load_json_file_from_timing(canonical_timing_path)
    canonical_input = timing.workloads[0].input_sha256
    canonical_cache = _cache_identity(timing.workloads[0].cache_clear)
    records = _load_replay_records(staging_dir)
    reasons = _replay_reasons(
        records,
        canonical_input,
        canonical_cache,
        artifact_paths,
    )
    reasons.extend(
        _environment_reasons(
            environment or [],
            expected_gpu_id=expected_gpu_id,
            expected_gpu_bdf=expected_gpu_bdf,
            expected_pcie_topology=expected_pcie_topology,
        ),
    )
    dispatch_digests = _dispatch_digests(
        counter_paths if counter_paths is not None else artifact_paths,
    )
    processes = [
        _process_evidence(record, dispatch_digests.get(record.pass_index))
        for record in records
        if record.pass_index in dispatch_digests
    ]
    if len(processes) != len(records):
        reasons.append("replay_dispatch_sequence_missing")
    if (
        processes
        and len({process.dispatch_sequence_digest for process in processes})
        != 1
    ):
        reasons.append("replay_dispatch_sequence_mismatch")
    alignment_digest = (
        stable_json_checksum(
            [
                process.model_dump(mode="json")
                for process in sorted(
                    processes, key=lambda item: item.pass_index
                )
            ],
        )
        if processes and not reasons
        else None
    )
    return PerformanceReplayEvidenceSidecar(
        status=(
            DiagnosticSidecarStatus.AVAILABLE
            if not reasons
            else DiagnosticSidecarStatus.PARTIAL
        ),
        run_id=run_id,
        candidate_sha256=candidate_sha256,
        canonical_input_sha256=canonical_input,
        processes=processes,
        environment=environment or [],
        artifact_sha256={
            path.name: sha256_file(path)
            for path in artifact_paths
            if path.is_file()
        },
        alignment_digest=alignment_digest,
        reason_codes=reasons,
    )


def load_json_file_from_timing(
    path: Path,
) -> PerformanceTimingEvidenceSidecar:
    """Load timing lazily to keep the replay contract dependency acyclic."""
    from sol_execbench.core.bench.performance_model.timing_evidence import (
        PerformanceTimingEvidenceSidecar,
    )

    timing = load_json_file(PerformanceTimingEvidenceSidecar, path)
    if len(timing.workloads) != 1:
        raise ValueError("replay evidence requires one timing workload")
    return timing


def _load_replay_records(staging_dir: Path) -> list[RawPerformanceReplayRecord]:
    records: list[RawPerformanceReplayRecord] = []
    for path in sorted(staging_dir.glob(RAW_REPLAY_GLOB)):
        records.extend(load_jsonl_file(RawPerformanceReplayRecord, path))
    return records


def _process_evidence(
    record: RawPerformanceReplayRecord,
    dispatch_sequence_digest: str | None,
) -> ReplayProcessEvidence:
    if dispatch_sequence_digest is None:
        raise ValueError("replay dispatch sequence digest is missing")
    payload = record.model_dump(mode="json", exclude={"protocol"})
    return ReplayProcessEvidence(
        **payload,
        dispatch_sequence_digest=dispatch_sequence_digest,
        process_fingerprint=stable_json_checksum(
            payload,
        ),
    )


def _dispatch_digests(paths: list[Path]) -> dict[int, str]:
    from sol_execbench.core.bench.rocm_profiler.counters import (
        counter_dispatch_sequence_digest,
        counter_pass_index,
    )

    result: dict[int, str] = {}
    for path in paths:
        pass_index = counter_pass_index(path)
        if pass_index is None or path.suffix.lower() != ".csv":
            continue
        if pass_index in result:
            raise ValueError("replay has multiple counter CSVs for one pass")
        result[pass_index] = counter_dispatch_sequence_digest(path)
    return result


def _replay_reasons(
    records: list[RawPerformanceReplayRecord],
    canonical_input: str,
    canonical_cache: str,
    artifacts: list[Path],
) -> list[str]:
    reasons: list[str] = []
    if not records:
        reasons.append("replay_process_evidence_missing")
    if any(record.input_sha256 != canonical_input for record in records):
        reasons.append("replay_input_sha256_mismatch")
    if any(
        record.cache_identity_sha256 != canonical_cache for record in records
    ):
        reasons.append("replay_cache_identity_canonical_mismatch")
    passes = [record.pass_index for record in records]
    if passes and sorted(passes) != list(range(1, len(passes) + 1)):
        reasons.append("replay_pass_sequence_invalid")
    if len(records) > MAX_REPLAY_PASSES:
        reasons.append("replay_pass_limit_exceeded")
    if (
        records
        and len({record.cache_identity_sha256 for record in records}) != 1
    ):
        reasons.append("replay_cache_identity_mismatch")
    if (
        records
        and len({tuple(record.marker_ranges) for record in records}) != 1
    ):
        reasons.append("replay_marker_sequence_mismatch")
    total_bytes = sum(
        path.stat().st_size for path in artifacts if path.is_file()
    )
    if total_bytes > MAX_REPLAY_BUNDLE_BYTES:
        reasons.append("replay_bundle_too_large")
    return list(dict.fromkeys(reasons))


def _cache_identity(cache: CacheClearEvidence | None) -> str:
    if cache is None:
        payload = {
            "detected_l2_bytes": None,
            "clear_buffer_bytes": None,
            "source": None,
            "fallback_reason": None,
        }
    else:
        payload = {
            "detected_l2_bytes": cache.detected_l2_bytes,
            "clear_buffer_bytes": cache.clear_buffer_bytes,
            "source": cache.source,
            "fallback_reason": cache.fallback_reason,
        }
    return stable_json_checksum(payload)


def _environment_reasons(
    snapshots: list[RuntimeGPUTelemetry],
    *,
    expected_gpu_id: str | None,
    expected_gpu_bdf: str | None,
    expected_pcie_topology: PCIeTopologyIdentity | None,
) -> list[str]:
    if {snapshot.phase for snapshot in snapshots} != {"pre", "post"}:
        return ["replay_environment_snapshot_incomplete"]
    reasons: list[str] = []
    for snapshot in snapshots:
        if expected_gpu_id is not None and snapshot.gpu_id != expected_gpu_id:
            reasons.append("replay_gpu_id_mismatch")
        if (
            expected_gpu_bdf is not None
            and snapshot.gpu_bdf != expected_gpu_bdf
        ):
            reasons.append("replay_gpu_bdf_mismatch")
        if snapshot.performance_level != "AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK":
            reasons.append("replay_clock_mode_unverified")
        if snapshot.foreign_process_count not in {0, None}:
            reasons.append("replay_foreign_gpu_process")
        if snapshot.temperature_c is None:
            reasons.append("replay_temperature_unavailable")
    topologies = [snapshot.pcie_topology for snapshot in snapshots]
    if any(topology is None for topology in topologies):
        reasons.append("replay_pcie_topology_incomplete")
    elif any(topology != topologies[0] for topology in topologies[1:]):
        reasons.append("replay_pcie_topology_changed")
    elif expected_pcie_topology is not None and (
        topologies[0] != expected_pcie_topology
    ):
        reasons.append("replay_pcie_topology_mismatch")
    return list(dict.fromkeys(reasons))


__all__ = [
    "MAX_DISPATCHES_PER_MARKER",
    "MAX_REPLAY_BUNDLE_BYTES",
    "MAX_REPLAY_PASSES",
    "RAW_REPLAY_GLOB",
    "REPLAY_EVIDENCE_ITERATIONS",
    "REPLAY_PROTOCOL_VERSION",
    "REPLAY_WARMUP_RUNS",
    "PerformanceReplayEvidenceSidecar",
    "RawPerformanceReplayRecord",
    "ReplayProtocol",
    "build_performance_replay_evidence",
]
