from __future__ import annotations

from typing import Any, cast

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.solution import BuildSpec, Solution
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity.schema_versions import (
    DEFINITION_SCHEMA_VERSION,
    SOLUTION_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    WORKLOAD_SCHEMA_VERSION,
)

JSONDict = dict[str, Any]


def json_dict(value: object) -> JSONDict:
    return cast(JSONDict, value)


def typed[T](value: object, typ: type[T]) -> T:
    del typ
    return cast(T, value)


def make_definition(**kwargs: Any) -> Definition:
    kwargs.setdefault("schema_version", DEFINITION_SCHEMA_VERSION)
    return Definition.model_validate(kwargs)


def make_workload(**kwargs: Any) -> Workload:
    kwargs.setdefault("schema_version", WORKLOAD_SCHEMA_VERSION)
    kwargs.setdefault(
        "checks",
        [{"type": "numeric", "output": "output"}],
    )
    return Workload.model_validate(kwargs)


def make_solution(**kwargs: Any) -> Solution:
    kwargs.setdefault("schema_version", SOLUTION_SCHEMA_VERSION)
    return Solution.model_validate(kwargs)


def make_build_spec(**kwargs: Any) -> BuildSpec:
    return BuildSpec.model_validate(kwargs)


def make_trace(**kwargs: Any) -> Trace:
    kwargs.setdefault("schema_version", TRACE_SCHEMA_VERSION)
    return Trace.model_validate(kwargs)
