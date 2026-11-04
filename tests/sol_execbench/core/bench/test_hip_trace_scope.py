# SPDX-License-Identifier: Apache-2.0
"""Independent-journal coverage, binding, and paired-loss rejection."""

import copy
import json

import pytest

from sol_execbench.core.bench.rocm_profiler.hip_scope import (
    qualify_hip_trace_scope,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_artifacts import (
    HIPScopeError,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    HIPScopeExpectation,
    HIPTraceScopeQualification,
)
from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.integrity.schema_versions import SchemaVersion

APIS = [
    "hipDeviceSynchronize",
    "hipLaunchKernel",
    "hipEventRecord",
    "hipStreamWaitEvent",
    "hipEventDestroy",
]


def row(kind, operation, identifier, stream, start, end):
    return {
        "kind": kind,
        "operation": operation,
        "correlation_id": {"internal": identifier},
        "stream_id": {"handle": stream},
        "thread_id": 7,
        "start_timestamp": start,
        "end_timestamp": end,
    }


@pytest.fixture
def evidence(tmp_path):
    binary = tmp_path / "probe"
    binary.write_bytes(b"audited synthetic executor fixture")
    command = (
        str(binary),
        "workload",
        str(tmp_path / "journal.json"),
        "a" * 32,
        sha256_file(binary),
        str(SchemaVersion.HIP_TRACE_JOURNAL),
    )
    expected = HIPScopeExpectation(
        run_id="a" * 32,
        workload_id="workload",
        candidate_sha256=sha256_file(binary),
        executor_source_sha256="c" * 64,
        device_uuid="GPU-0101010101010101",
        application_command=command,
        returncode=0,
    )
    calls, journal_calls = [], []
    specs = [
        (0, 0, 0, ""),
        (1, 1, 0, "producer"),
        (2, 1, 12, ""),
        (3, 2, 12, ""),
        (1, 2, 0, "consumer"),
        (0, 0, 0, ""),
        (4, 0, 12, ""),
    ]
    for i, (op, stream, event, kernel) in enumerate(specs, 1):
        call = row(0, op, i, stream, i * 100, i * 100 + 10)
        call.update(args=[], retval={"uint64_t_retval": 0})
        if stream:
            call["args"].append({"name": "stream", "value": hex(stream * 1000)})
        if event:
            call["args"].append({"name": "event", "value": hex(event)})
        if op == 3:
            call["args"].append({"name": "flags", "value": "0"})
        calls.append(call)
        journal_calls.append(
            {
                "name": APIS[op],
                "stream_handle": stream * 1000,
                "event_handle": event,
                "kernel_name": kernel,
                "result": 0,
            }
        )
    dispatches, barriers = [], []
    for identifier, stream, kernel in [(2, 1, 10), (5, 2, 11)]:
        item = row(
            2,
            0,
            identifier,
            stream,
            identifier * 100 + 15,
            identifier * 100 + 25,
        )
        item["dispatch_info"] = {
            "dispatch_id": identifier,
            "kernel_id": kernel,
            "agent_id": {"handle": 9},
            "queue_id": {"handle": stream},
        }
        dispatches.append(item)
    for identifier, stream, operation in [(3, 1, 0), (4, 2, 1)]:
        item = row(
            1,
            operation,
            identifier,
            stream,
            identifier * 100 + 15,
            identifier * 100 + 25,
        )
        item.update(
            agent_id={"handle": 9},
            queue_id={"handle": stream},
            source_queue_id={"handle": 1},
            hip_event_handle=12,
        )
        barriers.append(item)
    capture = {
        "metadata": {
            "pid": 42,
            "init_time": 1,
            "fini_time": 1000,
            "command": list(command),
        },
        "agents": [
            {
                "id": {"handle": 9},
                "uuid": {"bytes": {f"value{i}": int(i < 8) for i in range(16)}},
            }
        ],
        "kernel_symbols": [
            {"kernel_id": 10, "kernel_name": "producer.kd"},
            {"kernel_id": 11, "kernel_name": "consumer.kd"},
        ],
        "strings": {
            "buffer_records": [
                {"kind": "HIP_RUNTIME_API_EXT", "operations": APIS},
                {
                    "kind": "HIP_EVENT",
                    "operations": ["HIP_EVENT_RECORD", "HIP_EVENT_WAIT"],
                },
                {
                    "kind": "KERNEL_DISPATCH",
                    "operations": ["KERNEL_DISPATCH_COMPLETE"],
                },
            ]
        },
        "buffer_records": {
            "hip_api": calls,
            "kernel_dispatch": dispatches,
            "hip_event": barriers,
        },
    }
    journal = {
        "schema_version": SchemaVersion.HIP_TRACE_JOURNAL,
        "run_id": expected.run_id,
        "workload_id": expected.workload_id,
        "candidate_sha256": expected.candidate_sha256,
        "device_uuid": expected.device_uuid,
        "process_id": 42,
        "thread_id": 7,
        "completed": True,
        "correct": True,
        "calls": journal_calls,
    }
    return capture, journal, expected, binary


def qualify(tmp_path, evidence):
    capture, journal, expected, binary = evidence
    trace_path, journal_path = (
        tmp_path / "trace.json",
        tmp_path / "journal.json",
    )
    trace_path.write_text(json.dumps({"rocprofiler-sdk-tool": [capture]}))
    journal_path.write_text(json.dumps(journal))
    return qualify_hip_trace_scope(
        trace_path,
        journal_path,
        binary,
        expected,
        trace_sha256=sha256_file(trace_path),
        journal_sha256=sha256_file(journal_path),
    )


def test_all_dispatches_bound_to_controlled_scope(tmp_path, evidence):
    result = qualify(tmp_path, evidence)
    assert [d.dispatch_id for d in result.dispatches] == [2, 5]
    assert len(result.links) == 1
    assert result.scope.workload_id == "workload"
    assert not result.performance_authority and not result.schedule_complete
    assert (
        HIPTraceScopeQualification.model_validate_json(result.model_dump_json())
        == result
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "paired_record_loss",
        "paired_launch_loss",
        "extra_dispatch",
        "duplicate_dispatch",
        "swapped_calls",
        "wrong_kernel",
        "wrong_stream",
        "wrong_event",
        "wrong_agent",
        "wrong_pid",
        "wrong_thread",
        "wrong_run",
        "wrong_candidate",
        "wrong_device",
        "missing_end",
        "missing_fence",
        "failed_api",
        "outside_fence",
        "wrong_command",
        "wrong_schema",
        "wrong_binary",
        "journal_missing_call",
        "dispatch_missing",
    ],
    ids=str,
)
def test_missing_or_conflicting_evidence_never_qualifies(
    tmp_path, evidence, mutation
):
    capture, journal, _expected, binary = evidence
    records = capture["buffer_records"]
    calls, kernels, barriers = (
        records[k] for k in ("hip_api", "kernel_dispatch", "hip_event")
    )
    if mutation == "paired_record_loss":
        calls.pop(2)
        barriers.pop(0)
    elif mutation == "paired_launch_loss":
        calls.pop(1)
        kernels.pop(0)
    elif mutation in {"extra_dispatch", "duplicate_dispatch"}:
        extra = copy.deepcopy(kernels[0])
        if mutation == "extra_dispatch":
            extra["dispatch_info"]["dispatch_id"] = 999
        kernels.append(extra)
    elif mutation == "swapped_calls":
        calls[1]["start_timestamp"], calls[2]["start_timestamp"] = 300, 200
        calls[1]["end_timestamp"], calls[2]["end_timestamp"] = 310, 210
    elif mutation == "wrong_kernel":
        capture["kernel_symbols"][0]["kernel_name"] = "wrong"
    elif mutation == "wrong_stream":
        kernels[0]["stream_id"]["handle"] = 22
    elif mutation == "wrong_event":
        barriers[0]["hip_event_handle"] = 22
    elif mutation == "wrong_agent":
        kernels[0]["dispatch_info"]["agent_id"]["handle"] = 22
    elif mutation == "wrong_pid":
        journal["process_id"] += 1
    elif mutation == "wrong_thread":
        calls[1]["thread_id"] += 1
    elif mutation == "wrong_run":
        journal["run_id"] = "b" * 32
    elif mutation == "wrong_candidate":
        journal["candidate_sha256"] = "b" * 64
    elif mutation == "wrong_device":
        journal["device_uuid"] = "GPU-" + "b" * 16
    elif mutation == "missing_end":
        del journal["completed"]
    elif mutation == "missing_fence":
        calls.pop(5)
        journal["calls"].pop(5)
    elif mutation == "failed_api":
        calls[1]["retval"]["uint64_t_retval"] = 1
    elif mutation == "outside_fence":
        kernels[0]["end_timestamp"] = 700
    elif mutation == "wrong_command":
        capture["metadata"]["command"][3] = "b" * 32
    elif mutation == "wrong_schema":
        journal["schema_version"] = "unsupported"
    elif mutation == "wrong_binary":
        binary.write_bytes(b"changed executor")
    elif mutation == "journal_missing_call":
        journal["calls"].pop(1)
    else:
        kernels.pop(0)
    with pytest.raises(HIPScopeError):
        qualify(tmp_path, evidence)


def test_retained_digest_is_required_even_for_valid_json(tmp_path, evidence):
    qualify(tmp_path, evidence)
    trace = tmp_path / "trace.json"
    digest = sha256_file(trace)
    trace.write_text(trace.read_text() + "\n")
    with pytest.raises(HIPScopeError):
        qualify_hip_trace_scope(
            trace,
            tmp_path / "journal.json",
            evidence[3],
            evidence[2],
            trace_sha256=digest,
            journal_sha256=sha256_file(tmp_path / "journal.json"),
        )


def test_event_observation_tail_is_not_dispatch_completion_time(
    tmp_path, evidence
):
    evidence[0]["buffer_records"]["hip_event"][-1]["end_timestamp"] = 800
    assert qualify(tmp_path, evidence).status == "qualified_controlled_scope"
