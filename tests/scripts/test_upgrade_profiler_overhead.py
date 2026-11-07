# SPDX-License-Identifier: Apache-2.0
"""Incomplete profiler comparisons must not produce conclusions."""

from pathlib import Path

import pytest

from scripts.internal.upgrade.paired_analysis import BLOCKS
from scripts.internal.upgrade.profiler_overhead import Mode, command, report


def _rows() -> list[dict]:
    return [
        {
            "mode": mode,
            "block": block,
            "correct": True,
            "process_wall_ms": 2.0 if mode is Mode.WITH_EVENTS else 1.0,
            "workload_wall_ms": 1.0,
            "trace_bytes": 10,
        }
        for mode in Mode
        for block in range(BLOCKS)
    ]


def test_event_overhead_direction_is_cost_not_speedup() -> None:
    result = report(_rows())
    pair = result["comparisons"]["trace_with_events_over_trace_without_events"]
    assert pair["process_wall_ms"]["cost_ratio"] == pytest.approx(2.0)
    assert pair["workload_wall_ms"]["cost_ratio"] == 1.0


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "incorrect", "nonfinite"]
)
def test_invalid_observations_reject_whole_comparison(damage: str) -> None:
    rows = _rows()
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows.append(rows[0])
    elif damage == "incorrect":
        rows[0]["correct"] = False
    else:
        rows[0]["process_wall_ms"] = float("nan")
    with pytest.raises(ValueError, match=r"incomplete|invalid"):
        report(rows)


def test_disabled_events_override_runtime_trace_default(tmp_path: Path) -> None:
    arguments = command(Mode.WITHOUT_EVENTS, tmp_path)
    assert arguments[arguments.index("--hip-event-trace") + 1] == "false"
