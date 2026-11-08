"""Optional isolated-worker cost sidecar, outside production result contracts."""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from sol_execbench.core.data.base_model import CurrentSchemaModel
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.process.resource_guard import (
    WorkerResourceBudget,
    enforce_worker_budget,
)
from sol_execbench.core.solar_bridge.schema_versions import (
    CoverageArtifactSchema,
)
from solar.observation import collect_stage_costs


class StageObservation(CurrentSchemaModel):
    """Non-authoritative, directly observed costs for one worker."""

    current_schema_version = CoverageArtifactSchema.OBSERVATION
    schema_version: Literal[CoverageArtifactSchema.OBSERVATION] = (
        CoverageArtifactSchema.OBSERVATION
    )
    performance_authority: Literal[False] = False
    records: list[dict[str, object]]


@contextmanager
def worker_observation(path: Path | None) -> Iterator[None]:
    """Write observations even when an observed stage raises an exception."""
    if path is None:
        yield
        return
    records: list[dict[str, object]] = []
    try:
        with collect_stage_costs(records):
            yield
    finally:
        value = StageObservation(records=records)
        atomic_write_json_value(path, value.model_dump(mode="json"))


def add_worker_diagnostic_arguments(parser: argparse.ArgumentParser) -> None:
    """Register optional diagnostics without changing the worker IPC schema."""
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--memory-budget", type=int)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--evidence-budget", type=int)
    parser.add_argument("--budget-failure", type=Path)


@contextmanager
def worker_diagnostics(args: argparse.Namespace) -> Iterator[None]:
    """Enable explicitly supplied resource limits and stage observations."""
    budget = None
    if args.memory_budget is not None:
        if (
            args.evidence_root is None
            or args.evidence_budget is None
            or args.budget_failure is None
        ):
            raise ValueError("incomplete worker resource budget")
        budget = WorkerResourceBudget(
            memory_bytes=args.memory_budget,
            evidence_root=args.evidence_root,
            evidence_bytes=args.evidence_budget,
        )
    with (
        enforce_worker_budget(budget, args.budget_failure),
        worker_observation(args.observation),
    ):
        yield
