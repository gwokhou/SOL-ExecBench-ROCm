# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed three-stage readiness audit for the scored AKA corpus."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    atomic_write_jsonl_values,
    load_json_value,
)
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_contract import AkaCorpusRole
from sol_execbench.core.dataset.aka_corpus import (
    AkaCorpusEntry,
    AkaCorpusManifest,
)
from sol_execbench.core.integrity import (
    sha256_bytes,
    sha256_file,
    stable_json_checksum,
)
from sol_execbench.core.integrity.schema_versions import (
    CORPUS_STAGE_READINESS_RECORD_SCHEMA_VERSION,
    CORPUS_STAGE_READINESS_SUMMARY_SCHEMA_VERSION,
    CORPUS_STAGE_TRACE_IDENTITY_SCHEMA_VERSION,
)
from sol_execbench.core.process import exclusive_file_lock
from sol_execbench.core.solar_bridge.analyzer import (
    FORMAL_GFX_TARGET,
    formal_architecture_profile_hash,
)
from sol_execbench.core.solar_bridge.models import (
    READINESS_STAGES,
    SolarReadinessStatus,
    SolarStage,
    SolarStageAuditOutcome,
    SolarStageAuditRequest,
    SolarStageStatus,
)
from sol_execbench.core.solar_bridge.runner import run_solar_stage_worker
from sol_execbench.core.timestamps import utc_timestamp

_RESULT_FILENAME = "stage-result.json"
_MATRIX_FILENAME = "matrix.jsonl"
_SUMMARY_FILENAME = "summary.json"


class CorpusReadinessStatus(StrEnum):
    """Terminal states for a complete corpus readiness audit."""

    READY = "ready"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class CorpusStageAuditResult:
    """Paths and denominator counts for one complete corpus audit attempt."""

    status: CorpusReadinessStatus
    problems: int
    workloads: int
    extraction_passed: int
    conversion_passed: int
    verification_passed: int
    fully_ready_problems: int
    matrix_path: Path
    summary_path: Path

    def __post_init__(self) -> None:
        """Normalize constructor input and reject unknown corpus states."""
        object.__setattr__(self, "status", CorpusReadinessStatus(self.status))

    @property
    def ready(self) -> bool:
        """Return whether the complete corpus readiness audit passed."""
        return self.status is CorpusReadinessStatus.READY

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible corpus audit result."""
        value = asdict(self)
        value["status"] = self.status
        value["matrix_path"] = str(self.matrix_path)
        value["summary_path"] = str(self.summary_path)
        return value


@dataclass(frozen=True, slots=True)
class _CorpusAuditContext:
    """Immutable inputs shared by every workload in one corpus audit."""

    corpus: AkaCorpusManifest
    output: Path
    manifest_sha256: str
    architecture_sha256: str
    device: str
    timeout_seconds: float
    resume: bool


def audit_corpus_stage_readiness(
    manifest_path: Path,
    output_root: Path,
    *,
    device: str = "cuda:0",
    timeout_seconds: float = 14_400,
    resume: bool = False,
) -> CorpusStageAuditResult:
    """Audit every scored workload and publish a deterministic status matrix."""
    output = output_root.resolve()
    lock_path = output.parent / f".{output.name}.corpus-audit.lock"
    with exclusive_file_lock(lock_path):
        return _audit_corpus_stage_readiness_locked(
            manifest_path,
            output,
            device=device,
            timeout_seconds=timeout_seconds,
            resume=resume,
        )


def _audit_corpus_stage_readiness_locked(
    manifest_path: Path,
    output: Path,
    *,
    device: str,
    timeout_seconds: float,
    resume: bool,
) -> CorpusStageAuditResult:
    corpus = AkaCorpusManifest.load(manifest_path)
    if output.exists() and not resume:
        raise FileExistsError(
            f"corpus readiness output already exists: {output}",
        )
    output.mkdir(parents=True, exist_ok=True)
    context = _CorpusAuditContext(
        corpus=corpus,
        output=output,
        manifest_sha256=sha256_file(corpus.path),
        architecture_sha256=formal_architecture_profile_hash(),
        device=device,
        timeout_seconds=timeout_seconds,
        resume=resume,
    )
    records: list[dict[str, Any]] = []
    for entry in corpus.entries:
        if entry.role is not AkaCorpusRole.SCORED:
            continue
        records.extend(_audit_entry(context, entry))
    return _finish_audit(context, records)


def _audit_entry(
    context: _CorpusAuditContext,
    entry: AkaCorpusEntry,
) -> list[dict[str, Any]]:
    corpus = context.corpus
    problem_path = entry.relative_problem_dir.as_posix()
    problem_dir = corpus.authored_root / entry.relative_problem_dir
    definition = Definition.model_validate_json(
        (problem_dir / "definition.json").read_text(encoding="utf-8"),
    )
    workloads = _workloads(problem_dir / "workload.jsonl")
    records: list[dict[str, Any]] = []
    for workload_uuid in entry.workload_uuids:
        workload = workloads[workload_uuid]
        workload_output = (
            context.output
            / "workloads"
            / entry.relative_problem_dir
            / workload_uuid
        )
        result_path = workload_output / _RESULT_FILENAME
        identity = _identity(
            context,
            problem_path,
            definition,
            workload,
        )
        if context.resume and result_path.is_file():
            record = _load_resumed_record(result_path, context.output, identity)
            atomic_write_json_value(result_path, record)
        else:
            outcome = run_solar_stage_worker(
                SolarStageAuditRequest(
                    problem_dir=str(problem_dir.resolve()),
                    workload_uuid=workload_uuid,
                    output_dir=str(workload_output),
                    device=context.device,
                ),
                timeout_seconds=context.timeout_seconds,
            )
            record = _record(identity, outcome, context.output)
            atomic_write_json_value(result_path, record)
        records.append(record)
    return records


def _identity(
    context: _CorpusAuditContext,
    problem_path: str,
    definition: Definition,
    workload: Workload,
) -> dict[str, Any]:
    corpus = context.corpus
    problem_hashes = corpus.materialized_problem_sha256[problem_path]
    workload_payload = json.dumps(
        workload.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    identity = {
        "problem_path": problem_path,
        "workload_uuid": workload.uuid,
        "corpus_manifest_sha256": context.manifest_sha256,
        "definition_sha256": problem_hashes["definition_sha256"],
        "workload_file_sha256": problem_hashes["workload_sha256"],
        "workload_sha256": sha256_bytes(workload_payload),
        "reference_sha256": sha256_bytes(definition.reference.encode()),
        "gfx_target": FORMAL_GFX_TARGET,
        "architecture_sha256": context.architecture_sha256,
        "trace_seed": 200,
        "verification_seeds": [11, 29, 47],
        "verification_patterns": ["random", "zeros", "boundary"],
    }
    trace_contract = {
        "schema_version": CORPUS_STAGE_TRACE_IDENTITY_SCHEMA_VERSION,
        **{
            key: identity[key]
            for key in (
                "corpus_manifest_sha256",
                "definition_sha256",
                "workload_file_sha256",
                "workload_sha256",
                "reference_sha256",
                "gfx_target",
                "architecture_sha256",
                "trace_seed",
            )
        },
    }
    return {
        **identity,
        "trace_identity_sha256": stable_json_checksum(trace_contract),
    }


def _record(
    identity: dict[str, Any],
    outcome: SolarStageAuditOutcome,
    output: Path,
) -> dict[str, Any]:
    if (
        outcome.architecture_sha256 is not None
        and outcome.architecture_sha256 != identity["architecture_sha256"]
    ):
        raise ValueError("readiness worker architecture identity mismatch")
    ready = outcome.ready
    stages = [
        _rebase_stage(dict(stage), Path(outcome.output_dir or ""), output)
        for stage in outcome.stages
    ]
    record = {
        "schema_version": CORPUS_STAGE_READINESS_RECORD_SCHEMA_VERSION,
        **identity,
        "status": (
            SolarReadinessStatus.READY if ready else SolarReadinessStatus.FAILED
        ),
        "failure_stage": outcome.failure_stage,
        "reason_code": outcome.reason_code,
        "message": outcome.message,
        "stages": stages,
    }
    _verify_record_artifacts(record, output)
    return record


def _rebase_stage(
    stage: dict[str, Any],
    workload_output: Path,
    audit_root: Path,
) -> dict[str, Any]:
    artifact = stage.get("artifact")
    if not isinstance(artifact, dict):
        return stage
    artifact = dict(artifact)
    source = workload_output / str(artifact["path"])
    artifact["path"] = source.resolve().relative_to(audit_root).as_posix()
    stage["artifact"] = artifact
    return stage


def _load_resumed_record(
    result_path: Path,
    output: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    record = load_json_value(result_path)
    if not isinstance(record, dict):
        raise ValueError(f"invalid resumed readiness record: {result_path}")
    if (
        record.get("schema_version")
        != CORPUS_STAGE_READINESS_RECORD_SCHEMA_VERSION
    ):
        raise ValueError("resumed readiness record schema mismatch")
    if any(record.get(key) != value for key, value in identity.items()):
        raise ValueError("resumed readiness record identity mismatch")
    _verify_record_artifacts(record, output)
    if _record_ready(record):
        record["status"] = SolarReadinessStatus.READY
        record["failure_stage"] = None
        record["reason_code"] = None
        record["message"] = None
    return record


def _verify_record_artifacts(record: dict[str, Any], output: Path) -> None:
    for stage in record.get("stages") or []:
        artifact = stage.get("artifact") or {}
        if not artifact:
            continue
        path = output / str(artifact.get("path", ""))
        try:
            resolved = path.resolve()
            resolved.relative_to(output)
        except (OSError, ValueError) as exc:
            raise ValueError(
                "readiness artifact escapes the audit root",
            ) from exc
        if not resolved.is_file() or sha256_file(resolved) != artifact.get(
            "sha256",
        ):
            raise ValueError("readiness artifact identity mismatch")


def _record_ready(record: dict[str, Any]) -> bool:
    try:
        stages = {
            SolarStage(str(item.get("stage"))): item
            for item in record.get("stages") or []
        }
    except ValueError:
        return False
    if set(stages) != set(READINESS_STAGES):
        return False
    return all(
        item.get("status") == SolarStageStatus.PASSED
        and isinstance(item.get("artifact"), dict)
        and len(str(item["artifact"].get("sha256", ""))) == 64
        for item in stages.values()
    )


def _finish_audit(
    context: _CorpusAuditContext,
    records: list[dict[str, Any]],
) -> CorpusStageAuditResult:
    corpus = context.corpus
    expected = sum(
        len(entry.workload_uuids)
        for entry in corpus.entries
        if entry.role is AkaCorpusRole.SCORED
    )
    if not records or len(records) != expected:
        raise ValueError("corpus readiness workload denominator mismatch")
    matrix_path = context.output / _MATRIX_FILENAME
    atomic_write_jsonl_values(matrix_path, records)
    summary = _summary(context, records, matrix_path)
    summary_path = context.output / _SUMMARY_FILENAME
    atomic_write_json_value(summary_path, summary)
    counts = summary["stage_counts"]
    return CorpusStageAuditResult(
        status=CorpusReadinessStatus(str(summary["status"])),
        problems=int(summary["problem_count"]),
        workloads=int(summary["workload_count"]),
        extraction_passed=int(counts["graph_extraction"]),
        conversion_passed=int(counts["einsum_conversion"]),
        verification_passed=int(counts["conversion_verification"]),
        fully_ready_problems=int(summary["fully_ready_problem_count"]),
        matrix_path=matrix_path,
        summary_path=summary_path,
    )


def _summary(
    context: _CorpusAuditContext,
    records: list[dict[str, Any]],
    matrix_path: Path,
) -> dict[str, Any]:
    stage_counts = {
        stage: sum(
            any(
                item.get("stage") == stage
                and item.get("status") == SolarStageStatus.PASSED
                for item in record.get("stages") or []
            )
            for record in records
        )
        for stage in READINESS_STAGES
    }
    by_problem: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        by_problem[str(record["problem_path"])].append(
            record["status"] == SolarReadinessStatus.READY,
        )
    ready = all(_record_ready(record) for record in records)
    return {
        "schema_version": CORPUS_STAGE_READINESS_SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_timestamp(),
        "status": (
            CorpusReadinessStatus.READY
            if ready
            else CorpusReadinessStatus.INCOMPLETE
        ),
        "corpus_manifest_sha256": context.manifest_sha256,
        "gfx_target": FORMAL_GFX_TARGET,
        "problem_count": len(by_problem),
        "workload_count": len(records),
        "fully_ready_problem_count": sum(
            all(values) for values in by_problem.values()
        ),
        "stage_counts": stage_counts,
        "failure_counts": dict(
            sorted(
                Counter(
                    str(record.get("reason_code") or "unknown")
                    for record in records
                    if record["status"] != SolarReadinessStatus.READY
                ).items(),
            ),
        ),
        "matrix": {
            "path": _MATRIX_FILENAME,
            "sha256": sha256_file(matrix_path),
            "size_bytes": matrix_path.stat().st_size,
        },
    }


def _workloads(path: Path) -> dict[str, Workload]:
    values = [
        Workload.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {item.uuid: item for item in values}
    if len(result) != len(values):
        raise ValueError(f"duplicate workload UUID in {path}")
    return result


__all__ = ["CorpusStageAuditResult", "audit_corpus_stage_readiness"]
