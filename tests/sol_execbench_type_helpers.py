from __future__ import annotations

from typing import Any, TypeVar, cast

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.solution import BuildSpec, Solution
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload

JsonDict = dict[str, Any]

_T = TypeVar("_T")


def json_dict(value: object) -> JsonDict:
    return cast(JsonDict, value)


def typed(value: object, typ: type[_T]) -> _T:
    del typ
    return cast(_T, value)


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
