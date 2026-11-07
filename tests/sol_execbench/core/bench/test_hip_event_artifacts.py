# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""HIP event observations never turn timestamp coincidences into dependencies."""

import json
import sqlite3

import pytest

from sol_execbench.core.bench.agent_feedback.hip_events import (
    hip_event_feedback,
)
from sol_execbench.core.bench.decision.runtime import (
    runtime_decision_precedence,
)
from sol_execbench.core.bench.profile_summary.builder import (
    build_profile_summary_sidecar,
)
from sol_execbench.core.bench.rocm_profiler.hip_event_artifacts import (
    read_hip_events,
)
from sol_execbench.core.bench.rocm_profiler.hip_events import (
    HIPEventOperation,
    HIPEventRecord,
    normalize_event_generations,
)
from sol_execbench.core.bench.rocm_profiler.models import (
    Rocprofv3ArtifactKind,
    Rocprofv3ProfileArtifact,
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
)


def record(operation=HIPEventOperation.RECORD, **updates):
    values = {
        "capture": "trace:0",
        "process_id": 1,
        "agent_id": 2,
        "queue_id": 3,
        "source_queue_id": 3,
        "stream_id": 4,
        "event_handle": 5,
        "operation": operation,
        "start_ns": 10,
        "end_ns": 20,
    }
    values.update(updates)
    return HIPEventRecord.model_validate(values)


@pytest.mark.parametrize(
    "count", [1, 2], ids=["single-record", "reused-handle"]
)
def test_waits_remain_unknown_without_host_generation_identity(count):
    records = [
        record(start_ns=10 + i * 30, end_ns=20 + i * 30) for i in range(count)
    ]
    records.extend(
        [
            record(
                HIPEventOperation.WAIT, queue_id=6, start_ns=100, end_ns=110
            ),
            record(
                HIPEventOperation.WAIT, queue_id=7, start_ns=100, end_ns=110
            ),
        ]
    )
    result, warnings = normalize_event_generations(records)
    assert [r.generation for r in result[:count]] == list(range(1, count + 1))
    assert all(r.generation is None for r in result[count:])
    assert "hip_event_wait_generation_unverified" in warnings


def test_capture_process_and_device_isolation():
    records = [
        record(),
        record(process_id=2),
        record(agent_id=9),
        record(capture="other:0"),
    ]
    result, warnings = normalize_event_generations(records)
    assert [r.generation for r in result] == [1, 1, 1, 1]
    assert warnings == []


def test_duplicate_records_do_not_manufacture_generations():
    result, warnings = normalize_event_generations([record(), record()])
    assert all(r.generation is None for r in result)
    assert "hip_event_duplicate_or_ambiguous_record" in warnings


def test_sdk_json_uses_named_operation_table(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "rocprofiler-sdk-tool": [
                    {
                        "metadata": {"pid": 7},
                        "strings": {
                            "buffer_records": [
                                {
                                    "kind": "HIP_EVENT",
                                    "operations": [
                                        "HIP_EVENT_WAIT",
                                        "HIP_EVENT_RECORD",
                                    ],
                                }
                            ]
                        },
                        "buffer_records": {
                            "hip_event": [
                                {
                                    "kind": 0,
                                    "operation": 1,
                                    "agent_id": {"handle": 8},
                                    "queue_id": {"handle": 3},
                                    "source_queue_id": {"handle": 3},
                                    "hip_event_handle": 12,
                                    "start_timestamp": 10,
                                    "end_timestamp": 20,
                                }
                            ]
                        },
                    }
                ]
            }
        )
    )
    result, warnings = read_hip_events(path)
    assert warnings == []
    assert result[0].operation is HIPEventOperation.RECORD
    assert result[0].process_id == 7
    assert result[0].stream_id is None
    profile = Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.SUCCESS,
        profiler_available=True,
        command=("rocprofv3",),
        output_directory=tmp_path,
        output_file="trace",
        artifacts=(
            Rocprofv3ProfileArtifact(
                path=path,
                kind=Rocprofv3ArtifactKind.METADATA_JSON,
                size_bytes=path.stat().st_size,
            ),
        ),
    )
    sidecar = build_profile_summary_sidecar(profile_result=profile)
    assert sidecar.summary.hip_events == result
    assert (
        type(sidecar).model_validate(sidecar.model_dump(mode="json")) == sidecar
    )
    assert hip_event_feedback(profile)[0].severity == "info"
    assert not runtime_decision_precedence(profile).available


def test_rocpd_reads_only_supported_schema(tmp_path):
    path = tmp_path / "trace.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE rocpd_metadata(tag TEXT, value TEXT);
            INSERT INTO rocpd_metadata VALUES ('schema_version', '3.0.4');
            CREATE TABLE rocpd_string(id INTEGER, guid TEXT, string TEXT);
            INSERT INTO rocpd_string VALUES (1, 'capture', 'HIP_EVENT_RECORD');
            CREATE TABLE rocpd_hip_event(
                guid TEXT,nid INTEGER,pid INTEGER,agent_id INTEGER,queue_id INTEGER,
                stream_id INTEGER,hip_event_handle INTEGER,source_queue_id INTEGER,
                name_id INTEGER,start INTEGER,end INTEGER);
            INSERT INTO rocpd_hip_event VALUES ('capture',0,1,2,3,4,5,3,1,10,20);
        """)
    before = path.read_bytes()
    result, warnings = read_hip_events(path)
    assert len(result) == 1
    assert warnings == []
    assert path.read_bytes() == before
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE rocpd_metadata SET value='unknown'")
    assert read_hip_events(path)[0] == []


@pytest.mark.parametrize(
    "payload",
    ["{", "[]", '{"rocprofiler-sdk-tool": [null]}'],
    ids=["syntax", "root", "capture"],
)
def test_malformed_event_capture_is_diagnostic_only(tmp_path, payload):
    path = tmp_path / "trace.json"
    path.write_text(payload)
    records, warnings = read_hip_events(path)
    assert records == []
    assert warnings
