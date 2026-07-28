# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tracing-route profiles exposed by :mod:`solar.api`."""

from dataclasses import dataclass
from enum import StrEnum

from solar.graph.contracts import ExtractionKind


class Route(StrEnum):
    """Supported graph-tracing routes.

    A route selects only how the operator graph is captured.  The requested IR
    representation remains a separate choice so both routes can target the
    same NVLabs einsum IR without importing one another's implementation.
    """

    NVLABS = "nvlabs"
    MAINLINE = "mainline"


DEFAULT_ROUTE = Route.NVLABS


@dataclass(frozen=True)
class RouteSpec:
    """Declarative extraction profile for one route."""

    route: Route
    extraction: ExtractionKind


_ROUTE_SPECS = {
    Route.NVLABS: RouteSpec(
        route=Route.NVLABS,
        extraction=ExtractionKind.TORCHVIEW,
    ),
    Route.MAINLINE: RouteSpec(
        route=Route.MAINLINE,
        extraction=ExtractionKind.MAKE_FX_REFERENCE,
    ),
}


def normalize_route(value: Route | str) -> Route:
    """Return one supported route from a public option value."""
    try:
        return Route(value)
    except ValueError as exc:
        choices = ", ".join(route.value for route in Route)
        raise ValueError(
            f"unsupported SOLAR route {value!r}; choose: {choices}",
        ) from exc


def route_spec(value: Route | str) -> RouteSpec:
    """Return the declarative profile for one tracing route."""
    return _ROUTE_SPECS[normalize_route(value)]


__all__ = [
    "DEFAULT_ROUTE",
    "Route",
    "RouteSpec",
    "normalize_route",
    "route_spec",
]
