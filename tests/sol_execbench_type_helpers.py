from __future__ import annotations

from typing import Any, cast

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.schema_versions import BenchmarkArtifactSchema
from sol_execbench.core.data.solution import BuildSpec, Solution
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.platform.rdna4_validation import (
    HardwareValidationBinding,
)

JSONDict = dict[str, Any]


def make_hardware_validation_binding(
    *,
    source_revision: str = "a" * 40,
) -> HardwareValidationBinding:
    """Return a stable exact-SHA hardware binding for contract tests."""
    return HardwareValidationBinding(
        workflow_run_id=123,
        workflow_run_attempt=1,
        source_revision=source_revision,
        evidence_sha256="b" * 64,
        receipt_sha256="c" * 64,
        verified_at="2026-08-15T00:00:00Z",
    )


def json_dict(value: object) -> JSONDict:
    return cast(JSONDict, value)


def typed[T](value: object, typ: type[T]) -> T:
    del typ
    return cast(T, value)


def make_definition(**kwargs: Any) -> Definition:
    kwargs.setdefault("schema_version", BenchmarkArtifactSchema.DEFINITION)
    return Definition.model_validate(kwargs)


def make_workload(**kwargs: Any) -> Workload:
    kwargs.setdefault("schema_version", BenchmarkArtifactSchema.WORKLOAD)
    kwargs.setdefault(
        "checks",
        [{"type": "numeric", "output": "output"}],
    )
    return Workload.model_validate(kwargs)


def make_solution(**kwargs: Any) -> Solution:
    kwargs.setdefault("schema_version", BenchmarkArtifactSchema.SOLUTION)
    return Solution.model_validate(kwargs)


def make_build_spec(**kwargs: Any) -> BuildSpec:
    return BuildSpec.model_validate(kwargs)


def make_trace(**kwargs: Any) -> Trace:
    kwargs.setdefault("schema_version", BenchmarkArtifactSchema.TRACE)
    return Trace.model_validate(kwargs)
