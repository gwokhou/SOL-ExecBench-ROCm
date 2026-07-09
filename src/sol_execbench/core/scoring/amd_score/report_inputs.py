"""Input model parsing helpers for AMD-native score reports."""

from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload


def parse_score_report_definition(definition_payload: dict) -> Definition:
    """Parse a score report definition payload."""
    return Definition(**definition_payload)


def load_score_report_workloads(workload_path: Path) -> dict[str, Workload]:
    """Load workload records keyed by workload UUID."""
    return {
        workload.uuid: workload
        for workload in (
            Workload(**json.loads(line))
            for line in workload_path.read_text().splitlines()
            if line.strip()
        )
    }


def parse_score_report_trace(trace_payload: dict) -> Trace:
    """Parse one score report trace payload."""
    return Trace(**trace_payload)
