# SPDX-License-Identifier: Apache-2.0
"""Bounded event-tracing overhead screen, never a timing calibration."""

import argparse
import json
import math
import statistics
import sys
import time
from enum import StrEnum
from pathlib import Path

from scripts.internal.upgrade.paired_analysis import BLOCKS, summarize
from sol_execbench.core.bench.rocm_profiler.hip_event_artifacts import (
    read_hip_events,
)
from sol_execbench.core.bench.rocm_profiler.hip_events import HIPEventOperation
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

TIMEOUT_SECONDS = 60
CAPTURE_BYTES = 4096
PROBE = Path(__file__).with_name("event_probe.py")


class Mode(StrEnum):
    """Same workload, with exactly one event-tracing option changed."""

    NONE = "no_profile"
    WITHOUT_EVENTS = "trace_without_events"
    WITH_EVENTS = "trace_with_events"


def command(mode: Mode, root: Path) -> list[str]:
    """Construct explicit event options; runtime tracing enables them by default."""
    application = [sys.executable, str(PROBE)]
    if mode is Mode.NONE:
        return application
    return [
        "rocprofv3",
        "--hip-runtime-trace",
        "--kernel-trace",
        "--hip-event-trace",
        "true" if mode is Mode.WITH_EVENTS else "false",
        "--output-format",
        "json",
        "rocpd",
        "--output-directory",
        str(root),
        "--output-file",
        "probe",
        "--",
        *application,
    ]


def observe(mode: Mode, root: Path, block: int) -> dict:
    """Keep raw per-process and application timings and trace byte costs."""
    root.mkdir(parents=True)
    started = time.perf_counter()
    result = run_in_process_group_bounded(
        command(mode, root),
        timeout=TIMEOUT_SECONDS,
        max_capture_bytes=CAPTURE_BYTES,
    )
    elapsed = (time.perf_counter() - started) * 1000
    if result.returncode:
        raise RuntimeError(f"{mode} block {block} exited {result.returncode}")
    values = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.startswith('{"correct":')
    ]
    if len(values) != 1 or values[0].get("correct") is not True:
        raise ValueError("probe did not emit one correct workload receipt")
    artifacts = {
        str(p.relative_to(root)): {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }
        for p in root.rglob("*")
        if p.is_file()
    }
    event_count = 0
    record_count = 0
    wait_count = 0
    if mode is not Mode.NONE:
        traces = list(root.rglob("*.json"))
        if len(traces) != 1:
            raise ValueError("profiler must produce exactly one JSON trace")
        events, _ = read_hip_events(traces[0])
        event_count = len(events)
        record_count = sum(
            e.operation is HIPEventOperation.RECORD for e in events
        )
        wait_count = sum(e.operation is HIPEventOperation.WAIT for e in events)
        if mode is Mode.WITH_EVENTS and (record_count != 2 or wait_count > 4):
            raise ValueError("unexpected HIP event operation counts")
        if mode is Mode.WITHOUT_EVENTS and event_count:
            raise ValueError("disabled HIP event tracing emitted events")
    return {
        "mode": mode,
        "block": block,
        "process_wall_ms": elapsed,
        **values[0],
        "trace_bytes": sum(v["bytes"] for v in artifacts.values()),
        "artifacts": artifacts,
        "hip_event_count": event_count,
        "hip_event_record_count": record_count,
        "hip_event_wait_count": wait_count,
    }


def report(rows: list[dict]) -> dict:
    """Retain paired process variation rather than treating inner calls as trials."""
    indexed = {(row["mode"], row["block"]): row for row in rows}
    expected = {(mode, block) for mode in Mode for block in range(BLOCKS)}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("incomplete or duplicate profiler comparison")
    if any(
        row.get("correct") is not True
        or any(
            not math.isfinite(row[key]) or row[key] <= 0
            for key in ("process_wall_ms", "workload_wall_ms")
        )
        for row in rows
    ):
        raise ValueError("invalid timing or correctness result")
    comparisons = {}
    for mode in (Mode.WITHOUT_EVENTS, Mode.WITH_EVENTS):
        for baseline in (Mode.NONE, Mode.WITHOUT_EVENTS):
            if baseline is mode:
                continue
            key = f"{mode}_over_{baseline}"
            comparisons[key] = {}
            for metric in ("process_wall_ms", "workload_wall_ms"):
                ratios = [
                    math.log(
                        indexed[mode, block][metric]
                        / indexed[baseline, block][metric]
                    )
                    for block in range(BLOCKS)
                ]
                summary = summarize(ratios)
                comparisons[key][metric] = {
                    "cost_ratio": summary["old_over_new"],
                    "paired_95_percent_ci": summary["paired_95_percent_ci"],
                }
    return {
        "authority": "development_screening_only",
        "observations": rows,
        "comparisons": comparisons,
        "median_trace_bytes": {
            mode: statistics.median(
                row["trace_bytes"] for row in rows if row["mode"] == mode
            )
            for mode in Mode
        },
        "limitations": [
            "Natural clocks; no clock adjustment",
            "Single delayed stream-event fixture, not corpus-wide overhead",
            "Process timing includes startup and trace export",
            "Application timing includes intentional GPU delay",
            "No event-generation causal matching claim",
            "Observed GPU WAIT counts may be below four host waits; retained per trial",
        ],
    }


def main() -> None:
    """Run ten balanced pairs, preserving partial receipts on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    protocol = {
        "blocks": BLOCKS,
        "mode_order": "forward_even_reverse_odd",
        "runner_sha256": sha256_file(Path(__file__)),
        "probe_sha256": sha256_file(PROBE),
        "timeout_seconds": TIMEOUT_SECONDS,
    }
    atomic_write_json_value(args.output / "protocol.json", protocol)
    for block in range(BLOCKS):
        modes = tuple(Mode) if block % 2 == 0 else tuple(reversed(tuple(Mode)))
        for mode in modes:
            row = observe(mode, args.output / f"block-{block}" / mode, block)
            rows.append(row)
            atomic_write_json_value(args.output / "partial.json", rows)
    if (
        sha256_file(Path(__file__)) != protocol["runner_sha256"]
        or sha256_file(PROBE) != protocol["probe_sha256"]
    ):
        raise ValueError("probe source changed during acquisition")
    atomic_write_json_value(args.output / "report.json", report(rows))


if __name__ == "__main__":
    main()
