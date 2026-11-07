# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed authoring catalog for the AKA-derived benchmark corpus."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.dataset.aka_contract import (
    AKACapability,
    AKACorpusRole,
    AKAFusionDepth,
    AKAOperation,
    AKAPassKind,
    AKASourceFamily,
    AKASuite,
)


class AKASeedSpec(BaseModel):
    """One declarative problem specification in the AKA authoring catalog."""

    model_config = ConfigDict(frozen=True)

    name: str
    suite: AKASuite
    task_path: str
    op_type: AKAOperation
    dtype: DType
    pass_kind: AKAPassKind
    fusion_depth: AKAFusionDepth
    source_family: AKASourceFamily
    axes: dict[str, dict[str, object]]
    inputs: dict[str, dict[str, object]]
    outputs: dict[str, dict[str, object]]
    reference: str
    workloads: list[dict[str, object]]
    role: AKACorpusRole = AKACorpusRole.SCORED
    exclusion_reason_code: str = ""
    description: str = ""
    custom_inputs_entrypoint: str | None = None
    capabilities: tuple[AKACapability, ...] = ()


_AKA_SEED_SPECS = TypeAdapter(tuple[AKASeedSpec, ...])


def load_aka_seed_specs(path: Path) -> tuple[AKASeedSpec, ...]:
    """Load and validate an AKA authoring catalog without executing code."""
    return _AKA_SEED_SPECS.validate_json(path.read_bytes())


__all__ = ["AKASeedSpec", "load_aka_seed_specs"]
