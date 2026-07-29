# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Target-aware compatibility selection for the AKA-derived corpus."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any, TypedDict

from sol_execbench.core.bench.reference_protocol import (
    MAX_REFERENCE_TENSOR_STORAGE_BYTES,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.data.dtypes import dtype_storage_bits
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_contract import (
    AKACompatibilityStage,
    AKAProbeStatus,
    AKATargetGeneration,
)
from sol_execbench.core.platform.runtime import (
    CacheClearPolicy,
    RocmDeviceInfo,
    derive_cache_clear_policy,
)
from sol_execbench.core.process.logs import redacted_text_tail
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded


class AKAExecutionTargetSpec(TypedDict):
    """Static schema policy for one supported AKA execution target."""

    generation: AKATargetGeneration
    supported_tensor_dtypes: tuple[DType, ...]


AKA_EXECUTION_TARGET_SPECS: dict[str, AKAExecutionTargetSpec] = {
    "gfx942": {
        "generation": AKATargetGeneration.CDNA3,
        "supported_tensor_dtypes": (
            DType.BFLOAT16,
            DType.FLOAT16,
            DType.FLOAT32,
            DType.INT64,
            DType.INT32,
            DType.INT8,
            DType.UINT8,
            DType.BOOL,
        ),
    },
    "gfx1150": {
        "generation": AKATargetGeneration.RDNA3_5,
        "supported_tensor_dtypes": (
            DType.BFLOAT16,
            DType.FLOAT16,
            DType.FLOAT32,
            DType.FLOAT8_E4M3FN,
            DType.INT64,
            DType.INT32,
            DType.INT8,
            DType.UINT8,
            DType.BOOL,
        ),
    },
    "gfx1200": {
        "generation": AKATargetGeneration.RDNA4,
        "supported_tensor_dtypes": (
            DType.BFLOAT16,
            DType.FLOAT16,
            DType.FLOAT32,
            DType.FLOAT8_E4M3FN,
            DType.INT64,
            DType.INT32,
            DType.INT8,
            DType.UINT8,
            DType.BOOL,
        ),
    },
}
SUPPORTED_AKA_GFX_TARGETS = tuple(AKA_EXECUTION_TARGET_SPECS)
DEFAULT_PROBE_TIMEOUT_SECONDS = 120.0
PROBE_RESULT_PREFIX = "AKA_PROBE_RESULT="
_PROBE_CAPTURE_BYTES = 64 * 1024


class AKAProbeInfrastructureError(RuntimeError):
    """The compatibility probe could not produce workload-level evidence."""


@dataclass(frozen=True)
class AKAExecutionTarget:
    """Manifest-declared execution policy for one exact gfx target."""

    gfx_target: str
    generation: AKATargetGeneration
    supported_tensor_dtypes: frozenset[DType]

    def __post_init__(self) -> None:
        """Normalize manifest boundary values and reject unknown members."""
        object.__setattr__(
            self,
            "generation",
            AKATargetGeneration(self.generation),
        )
        object.__setattr__(
            self,
            "supported_tensor_dtypes",
            frozenset(DType(dtype) for dtype in self.supported_tensor_dtypes),
        )


@dataclass(frozen=True)
class AKAMaterializationTarget:
    """Observed target device used to select a materialized corpus."""

    device: RocmDeviceInfo
    cache_clear: CacheClearPolicy

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible materialization target."""
        return {
            "device": self.device.device,
            "device_index": self.device.index,
            "device_name": self.device.name,
            "gfx_target": self.device.gfx_target,
            "total_memory_bytes": self.device.total_memory_bytes,
            "l2_cache_bytes": self.device.l2_cache_bytes,
            "torch_version": self.device.torch_version,
            "hip_version": self.device.hip_version,
            "cache_clear": {
                "detected_l2_bytes": self.cache_clear.detected_l2_bytes,
                "clear_buffer_bytes": self.cache_clear.clear_buffer_bytes,
                "source": self.cache_clear.source,
                "fallback_reason": self.cache_clear.fallback_reason,
            },
        }


@dataclass(frozen=True)
class AKAWorkloadDecision:
    """Compatibility decision for one canonical workload."""

    problem_path: str
    workload_uuid: str
    included: bool
    stage: AKACompatibilityStage
    reason_code: str
    detail: str = ""
    metrics: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize caller input and reject unknown compatibility stages."""
        object.__setattr__(self, "stage", AKACompatibilityStage(self.stage))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible workload decision."""
        return {
            "path": self.problem_path,
            "workload_uuid": self.workload_uuid,
            "included": self.included,
            # PyYAML's safe dumper requires an exact built-in string here.
            "stage": str(self.stage),
            "reason_code": self.reason_code,
            "detail": self.detail,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class AKAProblemSelection:
    """Selected workloads for one corpus problem."""

    entry: Any
    workloads: tuple[Workload, ...]


@dataclass(frozen=True)
class AKACorpusSelection:
    """Complete target-specific partition of the canonical corpus."""

    problems: tuple[AKAProblemSelection, ...]
    decisions: tuple[AKAWorkloadDecision, ...]

    @property
    def excluded(self) -> tuple[AKAWorkloadDecision, ...]:
        """Return workloads excluded from the selected target."""
        return tuple(
            decision for decision in self.decisions if not decision.included
        )


@dataclass(frozen=True)
class StaticReferenceStorage:
    """Minimum contiguous storage required by one reference IPC case."""

    input_storage_bytes: int
    reference_case_bytes: int


Probe = Callable[
    [Path, int, Workload, AKAMaterializationTarget, float],
    AKAWorkloadDecision,
]


def load_execution_targets(
    payload: Mapping[str, Any],
) -> dict[str, AKAExecutionTarget]:
    """Parse the manifest's closed target execution catalog."""
    if set(payload) != set(SUPPORTED_AKA_GFX_TARGETS):
        raise ValueError(
            "AKA execution_targets must define exactly "
            + ", ".join(SUPPORTED_AKA_GFX_TARGETS),
        )
    targets: dict[str, AKAExecutionTarget] = {}
    for gfx_target, raw in payload.items():
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"AKA execution target {gfx_target} must be an object",
            )
        raw_dtypes = raw.get("supported_tensor_dtypes") or ()
        try:
            dtypes = frozenset(DType(str(value)) for value in raw_dtypes)
        except ValueError as exc:
            raise ValueError(
                f"AKA execution target {gfx_target} has unknown dtypes: "
                f"{sorted(map(str, raw_dtypes))}",
            ) from exc
        if not dtypes:
            raise ValueError(
                f"AKA execution target {gfx_target} lacks supported dtypes",
            )
        expected = AKA_EXECUTION_TARGET_SPECS[gfx_target]
        try:
            generation = AKATargetGeneration(str(raw.get("generation") or ""))
        except ValueError as exc:
            raise ValueError(
                f"AKA execution target {gfx_target} generation changed",
            ) from exc
        if generation != expected["generation"]:
            raise ValueError(
                f"AKA execution target {gfx_target} generation changed",
            )
        if dtypes != frozenset(expected["supported_tensor_dtypes"]):
            raise ValueError(
                f"AKA execution target {gfx_target} dtype policy changed",
            )
        targets[gfx_target] = AKAExecutionTarget(
            gfx_target=gfx_target,
            generation=generation,
            supported_tensor_dtypes=dtypes,
        )
    return targets


def materialization_target(device: RocmDeviceInfo) -> AKAMaterializationTarget:
    """Build target-selection evidence from an observed device."""
    if device.gfx_target not in SUPPORTED_AKA_GFX_TARGETS:
        raise ValueError(
            f"unsupported AKA execution target: {device.gfx_target}",
        )
    return AKAMaterializationTarget(
        device=device,
        cache_clear=derive_cache_clear_policy(device.l2_cache_bytes),
    )


def definition_tensor_dtypes(definition: Definition) -> frozenset[DType]:
    """Return every input/output tensor dtype required by a Definition."""
    tensors = [*definition.inputs.values(), *definition.outputs.values()]
    return frozenset(tensor.dtype for tensor in tensors)


def static_reference_storage(
    definition: Definition,
    workload: Workload,
) -> StaticReferenceStorage:
    """Compute reference IPC storage from schema shapes without allocating tensors."""
    input_shapes = definition.get_input_shapes(workload.axes)
    output_shapes = definition.get_output_shapes(workload.axes)
    input_bytes = _shaped_tensor_storage_bytes(definition.inputs, input_shapes)
    output_bytes = _shaped_tensor_storage_bytes(
        definition.outputs,
        output_shapes,
    )
    return StaticReferenceStorage(
        input_storage_bytes=input_bytes,
        reference_case_bytes=input_bytes + output_bytes,
    )


def _shaped_tensor_storage_bytes(
    specs: Mapping[str, Any],
    shapes: Mapping[str, tuple[int, ...] | None],
) -> int:
    return sum(
        (prod(shape) * dtype_storage_bits(specs[name].dtype) + 7) // 8
        for name, shape in shapes.items()
        if shape is not None
    )


def _static_exclusion(
    definition: Definition,
    workload: Workload,
    target: AKAExecutionTarget,
) -> tuple[str, str, dict[str, int]] | None:
    unsupported = sorted(
        definition_tensor_dtypes(definition) - target.supported_tensor_dtypes,
    )
    if unsupported:
        return (
            "unsupported_target_dtype",
            f"{target.gfx_target} does not support corpus dtype(s): {', '.join(unsupported)}",
            {},
        )
    storage = static_reference_storage(definition, workload)
    if storage.reference_case_bytes > MAX_REFERENCE_TENSOR_STORAGE_BYTES:
        return (
            "reference_ipc_payload_limit",
            "schema-derived reference case exceeds the trusted reference IPC limit",
            {
                "input_storage_bytes": storage.input_storage_bytes,
                "reference_case_bytes": storage.reference_case_bytes,
                "ipc_limit_bytes": MAX_REFERENCE_TENSOR_STORAGE_BYTES,
            },
        )
    return None


def select_corpus_for_target(
    *,
    authored_root: Path,
    entries: Iterable[Any],
    execution_target: AKAExecutionTarget,
    target: AKAMaterializationTarget,
    probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    probe: Probe | None = None,
) -> AKACorpusSelection:
    """Partition every canonical workload for one observed target."""
    if target.device.gfx_target != execution_target.gfx_target:
        raise ValueError(
            "observed device and manifest execution target do not match",
        )
    if probe_timeout_seconds <= 0:
        raise ValueError("AKA probe timeout must be positive")
    effective_probe = probe or probe_workload
    problems: list[AKAProblemSelection] = []
    decisions: list[AKAWorkloadDecision] = []
    for entry in entries:
        problem_dir = authored_root / entry.relative_problem_dir
        definition = Definition.model_validate_json(
            (problem_dir / "definition.json").read_text(encoding="utf-8"),
        )
        workloads = tuple(
            Workload.model_validate_json(line)
            for line in (problem_dir / "workload.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        selected: list[Workload] = []
        for row_index, workload in enumerate(workloads):
            static_exclusion = _static_exclusion(
                definition,
                workload,
                execution_target,
            )
            if static_exclusion is not None:
                reason_code, detail, metrics = static_exclusion
                decision = AKAWorkloadDecision(
                    problem_path=entry.relative_problem_dir.as_posix(),
                    workload_uuid=workload.uuid,
                    included=False,
                    stage=AKACompatibilityStage.STATIC,
                    reason_code=reason_code,
                    detail=detail,
                    metrics=metrics,
                )
            else:
                decision = effective_probe(
                    problem_dir,
                    row_index,
                    workload,
                    target,
                    probe_timeout_seconds,
                )
                expected_path = entry.relative_problem_dir.as_posix()
                if (
                    decision.problem_path != expected_path
                    or decision.workload_uuid != workload.uuid
                ):
                    raise AKAProbeInfrastructureError(
                        "probe decision identity does not match the selected workload",
                    )
            decisions.append(decision)
            if decision.included:
                selected.append(workload)
        if selected:
            problems.append(
                AKAProblemSelection(entry=entry, workloads=tuple(selected)),
            )
    return AKACorpusSelection(tuple(problems), tuple(decisions))


def _probe_command(
    problem_dir: Path,
    row_index: int,
    workload: Workload,
    target: AKAMaterializationTarget,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sol_execbench.core.dataset.aka_probe_worker",
        "--problem-dir",
        str(problem_dir),
        "--row-index",
        str(row_index),
        "--workload-uuid",
        workload.uuid,
        "--device",
        target.device.device,
        "--expected-arch",
        target.device.gfx_target,
    ]


def _parse_probe_output(
    stdout: str,
    *,
    problem_path: str,
    workload_uuid: str,
) -> AKAWorkloadDecision:
    lines = [
        line
        for line in stdout.splitlines()
        if line.startswith(PROBE_RESULT_PREFIX)
    ]
    if len(lines) != 1:
        raise AKAProbeInfrastructureError(
            f"probe worker returned {len(lines)} structured results for {workload_uuid}",
        )
    try:
        payload = json.loads(lines[0][len(PROBE_RESULT_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise AKAProbeInfrastructureError(
            f"probe worker returned invalid JSON for {workload_uuid}",
        ) from exc
    try:
        status = AKAProbeStatus(str(payload.get("status") or ""))
    except ValueError as exc:
        raise AKAProbeInfrastructureError(
            f"probe worker returned invalid status for {workload_uuid}",
        ) from exc
    if status is AKAProbeStatus.INFRASTRUCTURE_ERROR:
        raise AKAProbeInfrastructureError(
            str(payload.get("detail") or "probe failed"),
        )
    metrics = payload.get("metrics") or {}
    if not isinstance(metrics, dict) or any(
        not isinstance(value, int) for value in metrics.values()
    ):
        raise AKAProbeInfrastructureError(
            f"probe worker returned invalid metrics for {workload_uuid}",
        )
    included = status is AKAProbeStatus.COMPATIBLE
    return AKAWorkloadDecision(
        problem_path=problem_path,
        workload_uuid=workload_uuid,
        included=included,
        stage=AKACompatibilityStage.LIVE_PROBE,
        reason_code=str(
            payload.get("reason_code")
            or ("probe_passed" if included else "probe_failed"),
        ),
        detail=redacted_text_tail(str(payload.get("detail") or "")),
        metrics={str(key): value for key, value in metrics.items()},
    )


def probe_workload(
    problem_dir: Path,
    row_index: int,
    workload: Workload,
    target: AKAMaterializationTarget,
    timeout_seconds: float,
) -> AKAWorkloadDecision:
    """Run one target probe with bounded output and process-group cleanup."""
    command = _probe_command(problem_dir, row_index, workload, target)
    try:
        completed = run_in_process_group_bounded(
            command,
            cwd=problem_dir,
            timeout=timeout_seconds,
            max_capture_bytes=_PROBE_CAPTURE_BYTES,
        )
    except subprocess.TimeoutExpired:
        return AKAWorkloadDecision(
            problem_path=problem_dir.relative_to(
                problem_dir.parents[1],
            ).as_posix(),
            workload_uuid=workload.uuid,
            included=False,
            stage=AKACompatibilityStage.LIVE_PROBE,
            reason_code="probe_timeout",
            detail=f"probe exceeded {timeout_seconds:g} seconds",
        )
    if completed.returncode != 0:
        detail = redacted_text_tail(completed.stderr or completed.stdout or "")
        raise AKAProbeInfrastructureError(
            f"probe worker exited {completed.returncode} for {workload.uuid}: {detail}",
        )
    problem_path = problem_dir.relative_to(problem_dir.parents[1]).as_posix()
    return _parse_probe_output(
        completed.stdout or "",
        problem_path=problem_path,
        workload_uuid=workload.uuid,
    )


__all__ = [
    "AKA_EXECUTION_TARGET_SPECS",
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "PROBE_RESULT_PREFIX",
    "SUPPORTED_AKA_GFX_TARGETS",
    "AKACorpusSelection",
    "AKAExecutionTarget",
    "AKAMaterializationTarget",
    "AKAProbeInfrastructureError",
    "AKAWorkloadDecision",
    "StaticReferenceStorage",
    "load_execution_targets",
    "materialization_target",
    "probe_workload",
    "select_corpus_for_target",
    "static_reference_storage",
]
