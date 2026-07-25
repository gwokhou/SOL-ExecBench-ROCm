# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Outer-project models for crossing the SOLAR process boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping

FORMAL_BOUND_KIND = "capacity_constrained_tile_aware_v1"
FORMAL_ARTIFACT_PATHS = frozenset(
    {
        "operator_graph.yaml",
        "einsum_graph.yaml",
        "conversion-attestation.yaml",
        "solar-analysis.yaml",
    }
)


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
    publication_eligible: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SolarAnalysisOutcome":
        data = dict(value)
        data["artifacts"] = tuple(dict(item) for item in data.get("artifacts") or [])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifacts"] = list(self.artifacts)
        return data

    @property
    def is_formal_publication(self) -> bool:
        """Whether an analyzed worker response satisfies the formal contract."""
        if (
            self.status != "analyzed"
            or self.publication_eligible is not True
            or self.bound_kind != FORMAL_BOUND_KIND
            or self.output_dir is None
            or self.architecture_sha256 is None
            or len(self.architecture_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.architecture_sha256
            )
            or self.lower_bound_seconds is None
            or not math.isfinite(self.lower_bound_seconds)
            or self.lower_bound_seconds <= 0
        ):
            return False
        paths: set[str] = set()
        for artifact in self.artifacts:
            path = Path(str(artifact.get("path", "")))
            sha256 = str(artifact.get("sha256", ""))
            if (
                path.is_absolute()
                or ".." in path.parts
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                return False
            paths.add(str(path))
        return paths == FORMAL_ARTIFACT_PATHS


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
    "FORMAL_ARTIFACT_PATHS",
    "FORMAL_BOUND_KIND",
    "formal_precision_for_definition",
]
