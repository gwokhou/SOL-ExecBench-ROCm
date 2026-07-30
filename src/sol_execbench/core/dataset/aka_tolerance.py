# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Runtime-probed tolerance calibration for AKA-derived workloads."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

from pydantic import TypeAdapter

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import (
    OutputCheck,
    ToleranceSpec,
    Workload,
)
from sol_execbench.core.dataset.aka_contract import AKACorpusRole
from sol_execbench.core.integrity import (
    sha256_file,
    stable_json_checksum,
    validate_relative_artifact_path,
    validate_sha256,
)
from sol_execbench.core.integrity.schema_versions import (
    AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION,
)

CALIBRATION_METHOD = "repeated_reference_runs"
DEFAULT_MARGIN = 1.25
DEFAULT_SEED_COUNT = 3
DEFAULT_REPEATS_PER_SEED = 3
_REQUIRED_MATCHED_RATIO = 0.99
_MIN_ATOL_FLOOR = 1e-9

# Floors bound the candidate/reference comparison when a deterministic reference
# produces zero run-to-run variation. Observed variation can only widen them.
_DTYPE_FLOORS: dict[str, tuple[float, float]] = {
    "float64": (1e-9, 1e-9),
    "float32": (1e-5, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1e-2, 1e-2),
    "float8_e4m3fn": (1e-1, 1e-1),
    "float8_e5m2": (1e-1, 1e-1),
    "float4_e2m1": (5e-1, 5e-1),
    "float4_e2m1fn_x2": (5e-1, 5e-1),
    "int64": (0.0, 0.0),
    "int32": (0.0, 0.0),
    "int16": (0.0, 0.0),
    "int8": (0.0, 0.0),
    "bool": (0.0, 0.0),
}


class CalibrationStatus(StrEnum):
    """Runtime-probe disposition for one workload."""

    CALIBRATED = "calibrated"
    EXCLUDED = "excluded"


class CalibrationEntry(Protocol):
    """Corpus-entry fields needed to validate calibration coverage."""

    @property
    def role(self) -> AKACorpusRole: ...

    @property
    def exclusion_reason_code(self) -> str: ...

    @property
    def workload_uuids(self) -> tuple[str, ...]: ...

    @property
    def relative_problem_dir(self) -> Path: ...


def dtype_default_tolerance(
    dtype: str,
    *,
    margin: float = DEFAULT_MARGIN,
) -> ToleranceSpec:
    """Return the safety-margined floor for one output dtype."""
    atol, rtol = _DTYPE_FLOORS.get(dtype, _DTYPE_FLOORS["float32"])
    return _tolerance(
        max(_MIN_ATOL_FLOOR, atol * margin),
        rtol * margin,
    )


def calibrate_tolerance(
    output_dtypes: Sequence[str],
    *,
    observed_max_atol: float,
    observed_max_rtol: float,
    margin: float = DEFAULT_MARGIN,
) -> ToleranceSpec:
    """Combine per-workload runtime variation with dtype-safe floors."""
    if margin < 1.0:
        raise ValueError("calibration safety margin must be at least 1.0")
    if not output_dtypes:
        raise ValueError("calibration requires at least one output dtype")
    floors = [
        dtype_default_tolerance(dtype, margin=margin) for dtype in output_dtypes
    ]
    return _tolerance(
        max(
            _MIN_ATOL_FLOOR,
            observed_max_atol * margin,
            *(item.max_atol for item in floors),
        ),
        max(observed_max_rtol * margin, *(item.max_rtol for item in floors)),
    )


def _tolerance(max_atol: float, max_rtol: float) -> ToleranceSpec:
    return ToleranceSpec(
        max_atol=max_atol,
        max_rtol=max_rtol,
        required_matched_ratio=_REQUIRED_MATCHED_RATIO,
        max_error_cap=None,
        allow_negative_inf=False,
    )


def workload_contract_sha256(
    definition: Definition,
    workload: Workload,
) -> str:
    """Hash semantic Definition plus Workload fields excluding calibrated checks."""
    return stable_json_checksum(
        {
            "definition": definition.model_dump(mode="json"),
            "workload": workload.model_dump(
                mode="json",
                exclude={"checks"},
            ),
        },
    )


def load_tolerance_calibration(path: Path) -> dict[str, object]:
    """Load and validate the top-level shape of a calibration artifact."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("AKA tolerance calibration must be a JSON object")
    if (
        int(data.get("schema_version", 0))
        != AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION
    ):
        raise ValueError(
            f"AKA tolerance calibration must use schema_version "
            f"{AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION}",
        )
    if data.get("method") != CALIBRATION_METHOD:
        raise ValueError(
            f"AKA tolerance calibration method must be {CALIBRATION_METHOD}",
        )
    if float(data.get("margin", 0.0)) < 1.0:
        raise ValueError(
            "AKA tolerance calibration margin must be at least 1.0",
        )
    if int(data.get("seed_count", 0)) < 2:
        raise ValueError(
            "AKA tolerance calibration requires at least two seeds",
        )
    if int(data.get("repeats_per_seed", 0)) < 2:
        raise ValueError(
            "AKA tolerance calibration requires repeated executions",
        )
    if not isinstance(data.get("records"), list):
        raise ValueError("AKA tolerance calibration records must be a list")
    return data


def calibration_checks(path: Path) -> dict[str, list[OutputCheck]]:
    """Return calibrated output checks indexed by workload UUID."""
    data = load_tolerance_calibration(path)
    result: dict[str, list[OutputCheck]] = {}
    adapter = TypeAdapter(list[OutputCheck])
    for record in _record_list(data):
        if not isinstance(record, Mapping):
            raise ValueError(
                "AKA tolerance calibration record must be an object",
            )
        if record.get("status") != CalibrationStatus.CALIBRATED:
            continue
        uuid = str(record.get("workload_uuid") or "")
        if not uuid or uuid in result:
            raise ValueError(
                "calibration workload UUID is missing or duplicated",
            )
        result[uuid] = adapter.validate_python(record.get("checks") or [])
    return result


def validate_calibration_binding(
    *,
    authored_root: Path,
    binding: Mapping[str, object],
    entries: Sequence[CalibrationEntry],
    source_revision: str,
    formal_gfx_target: str,
) -> dict[str, object]:
    """Verify artifact identity, coverage, contracts, and authored tolerances."""
    relative = validate_relative_artifact_path(
        binding.get("path"),
        "tolerance calibration path",
    )
    expected_sha = validate_sha256(
        binding.get("sha256"),
        "tolerance calibration SHA-256",
    )
    path = authored_root / relative
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError("AKA tolerance calibration artifact identity changed")
    data = load_tolerance_calibration(path)
    if data.get("aka_revision") != source_revision:
        raise ValueError(
            "AKA tolerance calibration pins a different source revision",
        )
    device = data.get("device")
    if (
        not isinstance(device, Mapping)
        or device.get("gfx_target") != formal_gfx_target
    ):
        raise ValueError(
            "AKA tolerance calibration was not run on the formal target",
        )
    records = _index_records(_record_list(data))
    _validate_record_coverage(authored_root, entries, records)
    return {
        "path": relative,
        "sha256": expected_sha,
        "records": len(records),
        "method": data["method"],
        "gfx_target": formal_gfx_target,
    }


def _index_records(records: list[object]) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError(
                "AKA tolerance calibration record must be an object",
            )
        record = cast(Mapping[str, object], raw)
        uuid = str(record.get("workload_uuid") or "")
        if not uuid or uuid in indexed:
            raise ValueError(
                "calibration workload UUID is missing or duplicated",
            )
        indexed[uuid] = record
    return indexed


def _validate_record_coverage(
    authored_root: Path,
    entries: Sequence[CalibrationEntry],
    records: Mapping[str, Mapping[str, object]],
) -> None:
    expected_uuids = {
        uuid for entry in entries for uuid in tuple(entry.workload_uuids)
    }
    if set(records) != expected_uuids:
        raise ValueError(
            "AKA tolerance calibration workload coverage is incomplete",
        )
    for entry in entries:
        problem_path = entry.relative_problem_dir.as_posix()
        definition, workloads = _load_problem(authored_root / problem_path)
        for workload in workloads:
            _validate_workload_record(
                entry,
                problem_path,
                definition,
                workload,
                records[workload.uuid],
            )


def _load_problem(problem_dir: Path) -> tuple[Definition, tuple[Workload, ...]]:
    definition = Definition.model_validate_json(
        (problem_dir / "definition.json").read_text(encoding="utf-8"),
    )
    workloads = tuple(
        Workload.model_validate_json(line)
        for line in (problem_dir / "workload.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    )
    return definition, workloads


def _validate_workload_record(
    entry: CalibrationEntry,
    problem_path: str,
    definition: Definition,
    workload: Workload,
    record: Mapping[str, object],
) -> None:
    if record.get("problem_path") != problem_path:
        raise ValueError(
            f"calibration problem path mismatch for {workload.uuid}",
        )
    expected_status = (
        CalibrationStatus.EXCLUDED
        if entry.role == AKACorpusRole.TARGET_INCOMPATIBLE
        else CalibrationStatus.CALIBRATED
    )
    if record.get("status") != expected_status:
        raise ValueError(f"calibration status mismatch for {workload.uuid}")
    if record.get("contract_sha256") != workload_contract_sha256(
        definition,
        workload,
    ):
        raise ValueError(f"calibration contract changed for {workload.uuid}")
    if expected_status is CalibrationStatus.EXCLUDED:
        if record.get("reason_code") != entry.exclusion_reason_code:
            raise ValueError(
                f"calibration exclusion mismatch for {workload.uuid}",
            )
        return
    calibrated = TypeAdapter(list[OutputCheck]).validate_python(
        record.get("checks") or [],
    )
    if workload.checks != calibrated:
        raise ValueError(
            f"authored checks are not calibrated for {workload.uuid}",
        )
    samples = record.get("samples")
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError(
            f"calibration has no runtime samples for {workload.uuid}",
        )


def _record_list(data: Mapping[str, object]) -> list[object]:
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("AKA tolerance calibration records must be a list")
    return cast(list[object], records)


__all__ = [
    "CALIBRATION_METHOD",
    "DEFAULT_MARGIN",
    "DEFAULT_REPEATS_PER_SEED",
    "DEFAULT_SEED_COUNT",
    "CalibrationStatus",
    "calibrate_tolerance",
    "calibration_checks",
    "dtype_default_tolerance",
    "load_tolerance_calibration",
    "validate_calibration_binding",
    "workload_contract_sha256",
]
