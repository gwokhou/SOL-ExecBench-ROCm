# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Outer-project models for crossing the SOLAR process boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SolarWorkerRequest:
    problem_dir: str
    workload_uuid: str
    output_dir: str
    device: str
    orojenesis_home: str | None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SolarWorkerRequest":
        return cls(
            problem_dir=str(value["problem_dir"]),
            workload_uuid=str(value["workload_uuid"]),
            output_dir=str(value["output_dir"]),
            device=str(value["device"]),
            orojenesis_home=(
                str(value["orojenesis_home"]) if value.get("orojenesis_home") else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SolarAnalysisOutcome:
    status: str
    analysis_id: str
    output_dir: str | None = None
    architecture_sha256: str | None = None
    lower_bound_seconds: float | None = None
    bound_kind: str | None = None
    limiting_resource: str | None = None
    artifacts: tuple[dict[str, str], ...] = field(default_factory=tuple)
    stage: str | None = None
    reason_code: str | None = None
    message: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SolarAnalysisOutcome":
        data = dict(value)
        data["artifacts"] = tuple(dict(item) for item in data.get("artifacts") or [])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifacts"] = list(self.artifacts)
        return data


def formal_precision_for_definition(definition: Any) -> str:
    """Select SOLAR's fallback precision from an outer tensor contract."""
    dtypes = {
        str(spec.dtype).lower()
        for spec in (*definition.inputs.values(), *definition.outputs.values())
    }
    for aliases, precision in (
        (("float8", "fp8"), "fp8"),
        (("bfloat16", "bf16"), "bf16"),
        (("float16", "fp16"), "fp16"),
        (("float32", "fp32"), "fp32"),
    ):
        if any(any(alias in dtype for alias in aliases) for dtype in dtypes):
            return precision
    raise ValueError(f"formal SOLAR precision is unsupported: {sorted(dtypes)}")


__all__ = [
    "SolarAnalysisOutcome",
    "SolarWorkerRequest",
    "formal_precision_for_definition",
]
