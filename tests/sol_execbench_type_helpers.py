from __future__ import annotations

from typing import Any, cast

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.solution import BuildSpec, Solution
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload

JSONDict = dict[str, Any]


def json_dict(value: object) -> JSONDict:
    return cast(JSONDict, value)


def typed[T](value: object, typ: type[T]) -> T:
    del typ
    return cast(T, value)


def make_definition(**kwargs: Any) -> Definition:
    return Definition.model_validate(kwargs)


def make_workload(**kwargs: Any) -> Workload:
    return Workload.model_validate(kwargs)


def make_solution(**kwargs: Any) -> Solution:
    return Solution.model_validate(kwargs)


def make_build_spec(**kwargs: Any) -> BuildSpec:
    return BuildSpec.model_validate(kwargs)


def make_trace(**kwargs: Any) -> Trace:
    return Trace.model_validate(kwargs)
