# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Outer reporting seam for fixed-instance SOLAR exploration."""

from pathlib import Path

from solar.analysis.exploration_contracts import InverseRequest
from solar.explore import query_resources
from solar.pipeline.exploration import load_exploration_request


def inverse_report(path: Path) -> dict[str, object]:
    """Read a bounded pinned request and return its necessary resource constraints."""
    request = load_exploration_request(path, InverseRequest)
    return query_resources(request).model_dump(mode="json")
