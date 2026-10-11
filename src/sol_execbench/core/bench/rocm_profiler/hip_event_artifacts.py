# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded readers for ROCm 10.1 JSON and rocpd HIP event records."""

import json
import sqlite3
from pathlib import Path

from pydantic import ValidationError

from sol_execbench.core.bench.rocm_profiler.hip_events import (
    HIPEventOperation,
    HIPEventRecord,
    normalize_event_generations,
)

MAX_JSON_BYTES = 16_000_000
MAX_ROCPD_BYTES = 128_000_000
MAX_EVENT_RECORDS = 10_000
SQL_PROGRESS_INTERVAL = 1_000
SQL_PROGRESS_LIMIT = 10_000
ROCPD_EVENT_SCHEMA = "3.0.4"


def read_hip_events(path: Path) -> tuple[list[HIPEventRecord], list[str]]:
    """Read optional events, failing closed on malformed or truncated captures."""
    database = path.suffix in {".db", ".rocpd"}
    limit = MAX_ROCPD_BYTES if database else MAX_JSON_BYTES
    try:
        if path.stat().st_size > limit:
            return [], [f"{path.name}: hip_event_parse_byte_limit"]
        records = _read_rocpd(path) if database else _read_json(path)
        if len(records) > MAX_EVENT_RECORDS:
            return [], [f"{path.name}: hip_event_parse_row_limit"]
        normalized, warnings = normalize_event_generations(records)
        return normalized, [f"{path.name}: {warning}" for warning in warnings]
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        RecursionError,
        sqlite3.Error,
        ValidationError,
    ):
        return [], [f"{path.name}: hip_event_artifact_invalid_or_unsupported"]


def _read_json(path: Path) -> list[HIPEventRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("profile JSON is not an object")
    captures = payload.get("rocprofiler-sdk-tool", [])
    records = []
    for index, capture in enumerate(captures):
        if not isinstance(capture, dict):
            raise ValueError("profile capture is not an object")
        events = capture.get("buffer_records", {}).get("hip_event", [])
        if not events:
            continue
        metadata = capture["metadata"]
        kinds = capture["strings"]["buffer_records"]
        for event in events:
            kind = kinds[event["kind"]]
            if kind["kind"] != "HIP_EVENT":
                raise ValueError("incorrect event domain")
            records.append(
                HIPEventRecord(
                    capture=f"{path.name}:{index}",
                    process_id=metadata["pid"],
                    agent_id=event["agent_id"]["handle"],
                    queue_id=event["queue_id"]["handle"],
                    stream_id=event.get("stream_id", {}).get("handle"),
                    event_handle=event["hip_event_handle"],
                    source_queue_id=event["source_queue_id"]["handle"],
                    operation=HIPEventOperation(
                        kind["operations"][event["operation"]]
                    ),
                    start_ns=event["start_timestamp"],
                    end_ns=event["end_timestamp"],
                )
            )
            if len(records) > MAX_EVENT_RECORDS:
                return records
    return records


def _read_rocpd(path: Path) -> list[HIPEventRecord]:
    steps = 0

    def progress() -> int:
        nonlocal steps
        steps += 1
        return int(steps > SQL_PROGRESS_LIMIT)

    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.set_progress_handler(progress, SQL_PROGRESS_INTERVAL)
        versions = connection.execute(
            "SELECT value FROM rocpd_metadata WHERE tag='schema_version'"
        ).fetchall()
        if not versions or any(
            row[0] != ROCPD_EVENT_SCHEMA for row in versions
        ):
            raise ValueError("unsupported rocpd event schema")
        rows = connection.execute(
            "SELECT e.guid || ':' || e.nid,e.pid,e.agent_id,e.queue_id,e.stream_id,"
            "e.hip_event_handle,e.source_queue_id,s.string,e.start,e.end "
            "FROM rocpd_hip_event e LEFT JOIN rocpd_string s "
            "ON e.name_id=s.id AND e.guid=s.guid LIMIT ?",
            (MAX_EVENT_RECORDS + 1,),
        ).fetchall()
        return [
            HIPEventRecord(
                capture=f"{path.name}:{row[0]}",
                process_id=row[1],
                agent_id=row[2],
                queue_id=row[3],
                stream_id=row[4],
                event_handle=row[5],
                source_queue_id=row[6],
                operation=HIPEventOperation(row[7]),
                start_ns=row[8],
                end_ns=row[9],
            )
            for row in rows
        ]
    finally:
        connection.close()
