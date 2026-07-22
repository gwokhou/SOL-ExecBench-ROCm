# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Target-aware compatibility selection for the AKA-derived corpus."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypedDict

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.platform.runtime import (
    CacheClearPolicy,
    RocmDeviceInfo,
    derive_cache_clear_policy,
)
from sol_execbench.core.process.logs import redacted_text_tail
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded


class AkaExecutionTargetSpec(TypedDict):
    """Static schema policy for one supported AKA execution target."""

    generation: str
    supported_tensor_dtypes: tuple[str, ...]


AKA_EXECUTION_TARGET_SPECS: dict[str, AkaExecutionTargetSpec] = {
    "gfx942": {
        "generation": "cdna3",
        "supported_tensor_dtypes": ("bfloat16", "float16", "float32"),
    },
    "gfx1150": {
        "generation": "rdna3_5",
        "supported_tensor_dtypes": (
            "bfloat16",
            "float16",
            "float32",
            "float8_e4m3fn",
        ),
    },
    "gfx1200": {
        "generation": "rdna4",
        "supported_tensor_dtypes": (
            "bfloat16",
            "float16",
            "float32",
            "float8_e4m3fn",
        ),
    },
}
SUPPORTED_AKA_GFX_TARGETS = tuple(AKA_EXECUTION_TARGET_SPECS)
DEFAULT_PROBE_TIMEOUT_SECONDS = 120.0
PROBE_RESULT_PREFIX = "AKA_PROBE_RESULT="
_PROBE_CAPTURE_BYTES = 64 * 1024


class AkaProbeInfrastructureError(RuntimeError):
    """The compatibility probe could not produce workload-level evidence."""


@dataclass(frozen=True)
class AkaExecutionTarget:
    """Manifest-declared execution policy for one exact gfx target."""

    gfx_target: str
    generation: str
    supported_tensor_dtypes: frozenset[str]


@dataclass(frozen=True)
class AkaMaterializationTarget:
    """Observed target device used to select a materialized corpus."""

    device: RocmDeviceInfo
    cache_clear: CacheClearPolicy

    def to_dict(self) -> dict[str, Any]:
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
class AkaWorkloadDecision:
    """Compatibility decision for one canonical workload."""

    problem_path: str
    workload_uuid: str
    included: bool
    stage: str
    reason_code: str
    detail: str = ""
    metrics: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.problem_path,
            "workload_uuid": self.workload_uuid,
            "included": self.included,
            "stage": self.stage,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class AkaProblemSelection:
    """Selected workloads for one corpus problem."""

    entry: Any
    workloads: tuple[Workload, ...]


@dataclass(frozen=True)
class AkaCorpusSelection:
    """Complete target-specific partition of the canonical corpus."""

    problems: tuple[AkaProblemSelection, ...]
    decisions: tuple[AkaWorkloadDecision, ...]

    @property
    def excluded(self) -> tuple[AkaWorkloadDecision, ...]:
        return tuple(decision for decision in self.decisions if not decision.included)


Probe = Callable[
    [Path, int, Workload, AkaMaterializationTarget, float], AkaWorkloadDecision
]


def load_execution_targets(
    payload: Mapping[str, Any],
) -> dict[str, AkaExecutionTarget]:
    """Parse the manifest's closed target execution catalog."""

    if set(payload) != set(SUPPORTED_AKA_GFX_TARGETS):
        raise ValueError(
            "AKA execution_targets must define exactly "
            + ", ".join(SUPPORTED_AKA_GFX_TARGETS)
        )
    targets: dict[str, AkaExecutionTarget] = {}
    known_dtypes = {dtype.value for dtype in DType}
    for gfx_target, raw in payload.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"AKA execution target {gfx_target} must be an object")
        dtypes = frozenset(
            str(value) for value in raw.get("supported_tensor_dtypes") or ()
        )
        if not dtypes:
            raise ValueError(
                f"AKA execution target {gfx_target} lacks supported dtypes"
            )
        if not dtypes <= known_dtypes:
            raise ValueError(
                f"AKA execution target {gfx_target} has unknown dtypes: "
                f"{sorted(dtypes - known_dtypes)}"
            )
        expected = AKA_EXECUTION_TARGET_SPECS[gfx_target]
        generation = str(raw.get("generation") or "")
        if generation != expected["generation"]:
            raise ValueError(f"AKA execution target {gfx_target} generation changed")
        if dtypes != frozenset(expected["supported_tensor_dtypes"]):
            raise ValueError(f"AKA execution target {gfx_target} dtype policy changed")
        targets[gfx_target] = AkaExecutionTarget(
            gfx_target=gfx_target,
            generation=generation,
            supported_tensor_dtypes=dtypes,
        )
    return targets


def materialization_target(device: RocmDeviceInfo) -> AkaMaterializationTarget:
    """Build target-selection evidence from an observed device."""

    if device.gfx_target not in SUPPORTED_AKA_GFX_TARGETS:
        raise ValueError(f"unsupported AKA execution target: {device.gfx_target}")
    return AkaMaterializationTarget(
        device=device,
        cache_clear=derive_cache_clear_policy(device.l2_cache_bytes),
    )


def definition_tensor_dtypes(definition: Definition) -> frozenset[str]:
    """Return every input/output tensor dtype required by a Definition."""

    tensors = [*definition.inputs.values(), *definition.outputs.values()]
    return frozenset(tensor.dtype.value for tensor in tensors)


def _static_exclusion(
    definition: Definition,
    target: AkaExecutionTarget,
) -> tuple[str, str] | None:
    unsupported = sorted(
        definition_tensor_dtypes(definition) - target.supported_tensor_dtypes
    )
    if unsupported:
        return (
            "unsupported_target_dtype",
            f"{target.gfx_target} does not support corpus dtype(s): {', '.join(unsupported)}",
        )
    return None


def select_corpus_for_target(
    *,
    authored_root: Path,
    entries: Iterable[Any],
    execution_target: AkaExecutionTarget,
    target: AkaMaterializationTarget,
    probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    probe: Probe | None = None,
) -> AkaCorpusSelection:
    """Partition every canonical workload for one observed target."""

    if target.device.gfx_target != execution_target.gfx_target:
        raise ValueError("observed device and manifest execution target do not match")
    if probe_timeout_seconds <= 0:
        raise ValueError("AKA probe timeout must be positive")
    effective_probe = probe or probe_workload
    problems: list[AkaProblemSelection] = []
    decisions: list[AkaWorkloadDecision] = []
    for entry in entries:
        problem_dir = authored_root / entry.relative_problem_dir
        definition = Definition.model_validate_json(
            (problem_dir / "definition.json").read_text(encoding="utf-8")
        )
        workloads = tuple(
            Workload.model_validate_json(line)
            for line in (problem_dir / "workload.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        static_exclusion = _static_exclusion(definition, execution_target)
        selected: list[Workload] = []
        for row_index, workload in enumerate(workloads):
            if static_exclusion is not None:
                reason_code, detail = static_exclusion
                decision = AkaWorkloadDecision(
                    problem_path=entry.relative_problem_dir.as_posix(),
                    workload_uuid=workload.uuid,
                    included=False,
                    stage="static",
                    reason_code=reason_code,
                    detail=detail,
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
                    raise AkaProbeInfrastructureError(
                        "probe decision identity does not match the selected workload"
                    )
            decisions.append(decision)
            if decision.included:
                selected.append(workload)
        if selected:
            problems.append(AkaProblemSelection(entry=entry, workloads=tuple(selected)))
    return AkaCorpusSelection(tuple(problems), tuple(decisions))


def _probe_command(
    problem_dir: Path,
    row_index: int,
    workload: Workload,
    target: AkaMaterializationTarget,
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
    stdout: str, *, problem_path: str, workload_uuid: str
) -> AkaWorkloadDecision:
    lines = [
        line for line in stdout.splitlines() if line.startswith(PROBE_RESULT_PREFIX)
    ]
    if len(lines) != 1:
        raise AkaProbeInfrastructureError(
            f"probe worker returned {len(lines)} structured results for {workload_uuid}"
        )
    try:
        payload = json.loads(lines[0][len(PROBE_RESULT_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise AkaProbeInfrastructureError(
            f"probe worker returned invalid JSON for {workload_uuid}"
        ) from exc
    if payload.get("status") == "infrastructure_error":
        raise AkaProbeInfrastructureError(str(payload.get("detail") or "probe failed"))
    if payload.get("status") not in {"compatible", "incompatible"}:
        raise AkaProbeInfrastructureError(
            f"probe worker returned invalid status for {workload_uuid}"
        )
    metrics = payload.get("metrics") or {}
    if not isinstance(metrics, dict) or any(
        not isinstance(value, int) for value in metrics.values()
    ):
        raise AkaProbeInfrastructureError(
            f"probe worker returned invalid metrics for {workload_uuid}"
        )
    included = payload["status"] == "compatible"
    return AkaWorkloadDecision(
        problem_path=problem_path,
        workload_uuid=workload_uuid,
        included=included,
        stage="live_probe",
        reason_code=str(
            payload.get("reason_code")
            or ("probe_passed" if included else "probe_failed")
        ),
        detail=redacted_text_tail(str(payload.get("detail") or "")),
        metrics={str(key): value for key, value in metrics.items()},
    )


def probe_workload(
    problem_dir: Path,
    row_index: int,
    workload: Workload,
    target: AkaMaterializationTarget,
    timeout_seconds: float,
) -> AkaWorkloadDecision:
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
        return AkaWorkloadDecision(
            problem_path=problem_dir.relative_to(problem_dir.parents[1]).as_posix(),
            workload_uuid=workload.uuid,
            included=False,
            stage="live_probe",
            reason_code="probe_timeout",
            detail=f"probe exceeded {timeout_seconds:g} seconds",
        )
    if completed.returncode != 0:
        detail = redacted_text_tail(completed.stderr or completed.stdout or "")
        raise AkaProbeInfrastructureError(
            f"probe worker exited {completed.returncode} for {workload.uuid}: {detail}"
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
    "AkaCorpusSelection",
    "AkaExecutionTarget",
    "AkaMaterializationTarget",
    "AkaProbeInfrastructureError",
    "AkaWorkloadDecision",
    "load_execution_targets",
    "materialization_target",
    "probe_workload",
    "select_corpus_for_target",
]
