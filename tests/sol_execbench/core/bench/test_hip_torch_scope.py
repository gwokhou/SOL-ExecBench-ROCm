# SPDX-License-Identifier: Apache-2.0
"""Coverage, implicit dispatch and range binding must fail closed independently."""

import copy
import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.rocm_profiler.hip_scope_artifacts import (
    HIPScopeError,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_models import (
    HIPTorchExpectation,
    HIPTorchScope,
)
from sol_execbench.core.bench.rocm_profiler.hip_torch_scope import (
    qualify_torch_scope,
)
from sol_execbench.core.bench.rocm_profiler.schema_versions import (
    ProfilerArtifactSchema,
)
from sol_execbench.core.bench.rocm_profiler.scope_sources import (
    scope_runtime_digests,
)
from sol_execbench.core.integrity.checksums import sha256_file

APIS = [
    "hipDeviceSynchronize",
    "hipMemcpyAsync",
    "hipLaunchKernel",
    "hipEventRecord",
    "hipStreamWaitEvent",
    "hipGraphLaunch",
]


def activity(kind, operation, corr, stream, start, end):
    return {
        "kind": kind,
        "operation": operation,
        "correlation_id": {"internal": corr},
        "stream_id": {"handle": stream},
        "thread_id": 7,
        "start_timestamp": start,
        "end_timestamp": end,
    }


@pytest.fixture
def evidence(tmp_path):
    paths = {}
    for name in ("executor", "collector", "definition"):
        path = tmp_path / name
        path.write_text(f"audited {name}")
        paths[name + "_path"] = path
    expected = HIPTorchExpectation(
        run_id="a" * 32,
        workload_id="composite",
        device_uuid="GPU-0101010101010101",
        application_command=("python", "audited.py"),
        returncode=0,
        **{
            name + "_sha256": sha256_file(path)
            for name, path in (
                (key.removesuffix("_path"), value)
                for key, value in paths.items()
            )
        },
    )
    calls, events, dispatches, rows = [], [], [], []
    specs = [
        (1, 0, 0, 100),
        (2, 1, 9, 300),
        (3, 2, 9, 400),
        (4, 3, 9, 600),
        (5, 4, 10, 700),
        (6, 2, 10, 800),
        (7, 0, 0, 1000),
        (8, 0, 0, 1200),
    ]
    for corr, op, queue, start in specs:
        call = activity(0, op, corr, queue, start, start + 10)
        call.update(args=[], retval={"uint64_t_retval": 0})
        if op in (3, 4):
            call["args"].append({"name": "event", "value": "0xc"})
        if op == 4:
            call["args"].append({"name": "flags", "value": "0"})
        calls.append(call)
        if corr in (2, 3, 6):
            dispatch = activity(
                2, 0, corr, 0 if corr == 2 else queue, start + 15, start + 30
            )
            dispatch["dispatch_info"] = {
                "dispatch_id": corr,
                "kernel_id": corr,
                "agent_id": {"handle": 9},
                "queue_id": {"handle": queue},
            }
            dispatches.append(dispatch)
            rows.append(
                {
                    "kind": "enqueue",
                    "scope": 1,
                    "tid": 7,
                    "corr": corr,
                    "dispatch": corr,
                    "agent": 9,
                    "queue": queue,
                    "kernel": corr,
                }
            )
        if op in (3, 4):
            event = activity(1, op - 2, corr, queue, start + 15, start + 30)
            event.update(
                agent_id={"handle": 9},
                queue_id={"handle": queue},
                source_queue_id={"handle": 9},
                hip_event_handle=12,
            )
            events.append(event)
            rows.append(
                {
                    "kind": "event_enqueue",
                    "scope": 1,
                    "tid": 7,
                    "corr": corr,
                    "operation": op - 2,
                    "event": 12,
                    "agent": 9,
                    "queue": queue,
                    "source_queue": 9,
                }
            )
        rows.append(
            {
                "kind": "api",
                "scope": int(2 <= corr <= 7),
                "tid": 7,
                "corr": corr,
                "name": APIS[op],
                "result": 0,
                # Streamed launches and memcpys record the raw hipStream_t;
                # the consumer launch stays on the legacy default stream.
                **({"stream": 4660} if corr in (2, 3) else {}),
                **({"stream": 0} if corr == 6 else {}),
            }
        )
        if corr == 1:
            rows.append({"kind": "begin", "tid": 7, "timestamp": 200})
        if corr == 7:
            rows.append({"kind": "end", "tid": 7, "timestamp": 1100})
    for row in list(rows):
        if row["kind"] not in ("enqueue", "event_enqueue"):
            continue
        complete = {k: v for k, v in row.items() if k != "scope"}
        complete["kind"] = (
            "complete" if row["kind"] == "enqueue" else "event_complete"
        )
        start = next(s for c, _, _, s in specs if c == row["corr"])
        complete.update(start=start + 15, end=start + 30)
        rows.append(complete)
    rows.append({"kind": "seal", "tid": 7, "timestamp": 1300})
    for index, row in enumerate(rows, 1):
        row["seq"] = index
    capture = {
        "metadata": {
            "pid": 42,
            "init_time": 1,
            "fini_time": 1400,
            "command": list(expected.application_command),
        },
        "agents": [
            {
                "id": {"handle": 9},
                "uuid": {"bytes": {f"value{i}": int(i < 8) for i in range(16)}},
            }
        ],
        "kernel_symbols": [
            {"kernel_id": i, "kernel_name": name}
            for i, name in [(2, "copy_helper"), (3, "gemm"), (6, "consumer")]
        ],
        "strings": {
            "buffer_records": [
                {"kind": "HIP_RUNTIME_API_EXT", "operations": APIS},
                {
                    "kind": "HIP_EVENT",
                    "operations": [
                        "NONE",
                        "HIP_EVENT_RECORD",
                        "HIP_EVENT_WAIT",
                    ],
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
            "hip_event": events,
        },
    }
    ledger = {
        "schema_version": ProfilerArtifactSchema.HIP_SDK_LEDGER,
        "run_id": expected.run_id,
        "pid": 42,
        "records": rows,
        "row_count": len(rows),
        "drop_count": 0,
        "lossless": True,
        "finalized": True,
        "stop_ok": True,
        "flush_ok": True,
    }
    receipt = {
        key: getattr(expected, key)
        for key in (
            "run_id",
            "workload_id",
            "definition_sha256",
            "executor_sha256",
            "collector_sha256",
        )
    }
    receipt.update(
        process_id=42,
        device_uuid=str(UUID(bytes=b"0101010101010101")),
        shape=[32, 32, 32],
        correct=True,
        completed=True,
    )
    return capture, ledger, receipt, expected, paths


def qualify(tmp_path, evidence):
    capture, ledger, receipt, expected, paths = evidence
    payloads = {
        "torch_results.json": {"rocprofiler-sdk-tool": [capture]},
        "ledger.json": ledger,
        "workload.json": receipt,
    }
    for name, payload in payloads.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(json.dumps(payload))
    hashes = {name: sha256_file(tmp_path / name) for name in payloads}
    return qualify_torch_scope(tmp_path, expected, hashes, **paths)


def test_implicit_dispatch_is_bound_and_event_endpoints_remain_exact(
    tmp_path, evidence
):
    result = qualify(tmp_path, evidence)
    assert [r.dispatch_id for r in result.dispatches] == [2, 3, 6]
    assert result.implicit_dispatch_ids == [2]
    assert result.dispatches[0].stream_id == 0
    assert [
        (r.predecessor_dispatch_id, r.successor_dispatch_id)
        for r in result.links
    ] == [(3, 6)]
    assert not result.performance_authority
    assert not result.schedule_complete
    assert HIPTorchScope.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "mutation",
    [
        "paired_dispatch_loss",
        "missing_enqueue",
        "missing_completion",
        "export_and_buffer_loss",
        "export_and_enqueue_loss",
        "duplicate_dispatch",
        "missing_implicit",
        "scope_retag",
        "shift_boundary",
        "missing_footer",
        "dropped",
        "flush_failed",
        "stop_failed",
        "schema",
        "sequence_gap",
        "wrong_pid",
        "wrong_run",
        "wrong_workload",
        "wrong_device",
        "wrong_thread",
        "wrong_candidate",
        "wrong_command",
        "outside_gpu_fence",
        "missing_sync",
        "failed_sync",
        "failed_correctness",
        "graph",
        "paired_event_loss",
        "event_handle",
        "event_flags",
        "event_source",
        "noop_wait",
        "event_completion_timestamp",
        "overlap_host_calls",
    ],
)
def test_incomplete_or_misbound_scope_rejected(tmp_path, evidence, mutation):
    capture, ledger, receipt, expected, paths = copy.deepcopy(evidence)
    records, rows = capture["buffer_records"], ledger["records"]
    if mutation in {"paired_dispatch_loss", "missing_implicit"}:
        target = 2 if mutation == "missing_implicit" else 3
        records["kernel_dispatch"] = [
            r
            for r in records["kernel_dispatch"]
            if r["dispatch_info"]["dispatch_id"] != target
        ]
        records["hip_api"] = [
            r
            for r in records["hip_api"]
            if r["correlation_id"]["internal"] != target
        ]
    elif mutation in {"missing_enqueue", "missing_completion"}:
        kind = "enqueue" if mutation == "missing_enqueue" else "complete"
        rows.remove(next(r for r in rows if r["kind"] == kind))
        for index, row in enumerate(rows, 1):
            row["seq"] = index
        ledger["row_count"] = len(rows)
    elif mutation in {"export_and_buffer_loss", "export_and_enqueue_loss"}:
        kind = "complete" if mutation == "export_and_buffer_loss" else "enqueue"
        rows.remove(
            next(r for r in rows if r["kind"] == kind and r["dispatch"] == 2)
        )
        records["kernel_dispatch"].pop(0)
        for index, row in enumerate(rows, 1):
            row["seq"] = index
        ledger["row_count"] = len(rows)
    elif mutation == "duplicate_dispatch":
        records["kernel_dispatch"].append(
            copy.deepcopy(records["kernel_dispatch"][0])
        )
    elif mutation == "scope_retag":
        next(r for r in rows if r["kind"] == "enqueue")["scope"] = 0
    elif mutation == "shift_boundary":
        next(r for r in rows if r["kind"] == "begin")["timestamp"] = 500
    elif mutation == "missing_footer":
        del ledger["finalized"]
    elif mutation in {
        "dropped",
        "flush_failed",
        "stop_failed",
        "schema",
        "wrong_pid",
        "wrong_run",
    }:
        key, value = {
            "dropped": ("drop_count", 1),
            "flush_failed": ("flush_ok", False),
            "stop_failed": ("stop_ok", False),
            "schema": ("schema_version", "invalid"),
            "wrong_pid": ("pid", 43),
            "wrong_run": ("run_id", "b" * 32),
        }[mutation]
        ledger[key] = value
    elif mutation == "sequence_gap":
        rows[1]["seq"] += 1
    elif mutation in {
        "wrong_workload",
        "wrong_device",
        "wrong_candidate",
        "failed_correctness",
    }:
        key, value = {
            "wrong_workload": ("workload_id", "other"),
            "wrong_device": (
                "device_uuid",
                str(UUID(bytes=b"0202020202020202")),
            ),
            "wrong_candidate": ("executor_sha256", "b" * 64),
            "failed_correctness": ("correct", False),
        }[mutation]
        receipt[key] = value
    elif mutation == "wrong_thread":
        next(r for r in rows if r["kind"] == "enqueue")["tid"] += 1
    elif mutation == "wrong_command":
        capture["metadata"]["command"] = ["other"]
    elif mutation == "outside_gpu_fence":
        records["kernel_dispatch"][0]["end_timestamp"] = 1150
        next(r for r in rows if r["kind"] == "complete")["end"] = 1150
    elif mutation in {"missing_sync", "failed_sync"}:
        if mutation == "missing_sync":
            records["hip_api"] = [
                r
                for r in records["hip_api"]
                if r["correlation_id"]["internal"] != 7
            ]
        else:
            records["hip_api"][6]["retval"]["uint64_t_retval"] = 1
    elif mutation == "graph":
        records["kernel_dispatch"][0]["graph_exec_id"] = 1
    elif mutation in {"paired_event_loss", "noop_wait"}:
        records["hip_event"].pop()
        if mutation == "paired_event_loss":
            records["hip_api"] = [
                r
                for r in records["hip_api"]
                if r["correlation_id"]["internal"] != 5
            ]
        else:
            rows[:] = [
                r
                for r in rows
                if not (r["kind"].startswith("event_") and r["corr"] == 5)
            ]
            for index, row in enumerate(rows, 1):
                row["seq"] = index
            ledger["row_count"] = len(rows)
    elif mutation == "event_handle":
        records["hip_api"][4]["args"][0]["value"] = "0xd"
    elif mutation == "event_flags":
        records["hip_api"][4]["args"][1]["value"] = "1"
    elif mutation == "event_source":
        records["hip_event"][1]["source_queue_id"]["handle"] = 99
    elif mutation == "event_completion_timestamp":
        records["hip_event"][1]["end_timestamp"] += 10
    elif mutation == "overlap_host_calls":
        records["hip_api"][3]["start_timestamp"] = 305
    with pytest.raises((HIPScopeError, ValidationError)):
        qualify(tmp_path, (capture, ledger, receipt, expected, paths))


def test_evidence_digest_is_required(tmp_path, evidence):
    qualify(tmp_path, evidence)
    _, _, _, expected, paths = evidence
    hashes = {
        name: sha256_file(tmp_path / name)
        for name in ("torch_results.json", "ledger.json", "workload.json")
    }
    (tmp_path / "ledger.json").write_text("{}")
    with pytest.raises(HIPScopeError):
        qualify_torch_scope(tmp_path, expected, hashes, **paths)


def test_implicit_helper_can_be_the_actual_event_predecessor(
    tmp_path, evidence
):
    capture, ledger, receipt, expected, paths = copy.deepcopy(evidence)
    capture["buffer_records"]["hip_api"][2]["operation"] = 1
    next(r for r in ledger["records"] if r["kind"] == "api" and r["corr"] == 3)[
        "name"
    ] = "hipMemcpyAsync"
    result = qualify(tmp_path, (capture, ledger, receipt, expected, paths))
    assert result.implicit_dispatch_ids == [2, 3]
    assert result.links[0].predecessor_dispatch_id == 3


def test_default_stream_mixing_blocks_schedule_completeness(tmp_path, evidence):
    capture, ledger, receipt, expected, paths = copy.deepcopy(evidence)
    result = qualify(tmp_path, (capture, ledger, receipt, expected, paths))
    assert not result.schedule_complete
    assert result.schedule.default_stream_present
    assert result.schedule.lane_count == 3
    assert "default_stream_implicit_sync_not_inferred" in map(
        str, result.schedule.reason_codes
    )
    assert result.copy_domain.qualified


def test_explicit_streams_only_schedule_completes(tmp_path, evidence):
    capture, ledger, receipt, expected, paths = copy.deepcopy(evidence)
    next(
        r
        for r in capture["buffer_records"]["kernel_dispatch"]
        if r["correlation_id"]["internal"] == 2
    )["stream_id"]["handle"] = 9
    # Move the consumer launch off the legacy default stream as well.
    next(
        r
        for r in ledger["records"]
        if r["kind"] == "api" and r.get("corr") == 6
    )["stream"] = 4661
    result = qualify(tmp_path, (capture, ledger, receipt, expected, paths))
    assert result.schedule_complete
    assert not result.schedule.reason_codes
    assert not result.performance_authority
    assert [
        (r.predecessor_dispatch_id, r.successor_dispatch_id)
        for r in result.links
    ] == [(3, 6)]


def _insert_engine_copy(capture, ledger):
    """Add one in-scope D2D engine copy on queue 9 just before the record."""
    ledger["records"].append(
        {
            "kind": "api",
            "scope": 1,
            "tid": 7,
            "corr": 12,
            "name": "hipMemcpyAsync",
            "result": 0,
            "stream": 153,
        }
    )
    ledger["records"].append(
        {
            "kind": "memcpy",
            "tid": 7,
            "corr": 12,
            "operation": 4,
            "src_agent": 9,
            "dst_agent": 9,
            "bytes": 4096,
            "start": 455,
            "end": 470,
        }
    )
    capture["strings"]["buffer_records"].append(
        {
            "kind": "MEMORY_COPY",
            "operations": [
                "MEMORY_COPY_NONE",
                "MEMORY_COPY_HOST_TO_HOST",
                "MEMORY_COPY_HOST_TO_DEVICE",
                "MEMORY_COPY_DEVICE_TO_HOST",
                "MEMORY_COPY_DEVICE_TO_DEVICE",
            ],
        }
    )
    capture["buffer_records"]["hip_api"].append(
        {
            "kind": 0,
            "operation": 1,
            "correlation_id": {"internal": 12},
            "stream_id": {"handle": 9},
            "thread_id": 7,
            "start_timestamp": 451,
            "end_timestamp": 452,
            "args": [{"name": "stream", "value": "0x99"}],
            "retval": {"uint64_t_retval": 0},
        }
    )
    capture["buffer_records"]["memory_copy"] = [
        {
            "kind": 3,
            "operation": 4,
            "correlation_id": {"internal": 12},
            "thread_id": 7,
            "start_timestamp": 455,
            "end_timestamp": 470,
            "src_agent_id": {"handle": 9},
            "dst_agent_id": {"handle": 9},
            "bytes": 4096,
        }
    ]


def test_engine_copy_is_the_actual_event_predecessor(tmp_path, evidence):
    capture, ledger, receipt, expected, paths = copy.deepcopy(evidence)
    capture["buffer_records"]["memory_copy"] = []
    _insert_engine_copy(capture, ledger)
    rows = ledger["records"]
    copy_api = next(
        r for r in rows if r.get("corr") == 12 and r["kind"] == "api"
    )
    record_row = next(
        r for r in rows if r["kind"] == "event_enqueue" and r["corr"] == 4
    )
    rows.remove(copy_api)
    rows.insert(
        rows.index(
            next(r for r in rows if r["kind"] == "api" and r["corr"] == 3)
        )
        + 1,
        copy_api,
    )
    memcpy_row = next(r for r in rows if r["kind"] == "memcpy")
    rows.remove(memcpy_row)
    rows.insert(rows.index(copy_api) + 1, memcpy_row)
    for index, row in enumerate(rows, 1):
        row["seq"] = index
    ledger["row_count"] = len(rows)
    assert record_row["seq"] > copy_api["seq"]
    result = qualify(tmp_path, (capture, ledger, receipt, expected, paths))
    assert result.copy_domain.qualified
    assert result.copy_domain.copies[0].queue_id == 9
    link = result.links[0]
    assert link.predecessor_copy_correlation_id == 12
    assert link.predecessor_dispatch_id is None
    assert link.successor_dispatch_id == 6
    assert not result.schedule_complete


def test_no_event_workload_has_coverage_without_invented_links(
    tmp_path, evidence
):
    capture, ledger, receipt, expected, paths = copy.deepcopy(evidence)
    records = capture["buffer_records"]
    records["hip_event"] = []
    records["hip_api"] = [
        r
        for r in records["hip_api"]
        if r["correlation_id"]["internal"] not in (4, 5)
    ]
    ledger["records"] = [
        r for r in ledger["records"] if r.get("corr") not in (4, 5)
    ]
    for index, row in enumerate(ledger["records"], 1):
        row["seq"] = index
    ledger["row_count"] = len(ledger["records"])
    result = qualify(tmp_path, (capture, ledger, receipt, expected, paths))
    assert not result.links
    assert len(result.dispatches) == 3


@pytest.fixture
def production_evidence(tmp_path, evidence):
    from sol_execbench.core.bench.rocm_profiler.evaluate_scope_models import (
        SCOPE_LEDGER_FILE,
        SCOPE_RECEIPT_FILE,
        SCOPE_TRACE_FILE,
        EvaluateScopeIntent,
        EvaluateScopeReceipt,
    )
    from sol_execbench.core.integrity.checksums import stable_json_checksum

    capture, ledger, receipt, expected, paths = evidence
    intent = EvaluateScopeIntent(
        run_id=expected.run_id,
        workload_uuid="test-workload",
        workload_sha256="1" * 64,
        candidate_sha256="2" * 64,
        definition_sha256=expected.definition_sha256,
        device_uuid=expected.device_uuid,
        application_command=expected.application_command,
        staged_sha256={"executor": expected.executor_sha256},
        runtime_sha256=scope_runtime_digests(),
        collector_sha256=expected.collector_sha256,
    )
    bound = EvaluateScopeReceipt(
        intent_sha256=stable_json_checksum(intent.model_dump(mode="json")),
        input_sha256="3" * 64,
        process_id=42,
        device_uuid=receipt["device_uuid"],
        correct=True,
        completed=True,
    )
    payloads = {
        SCOPE_TRACE_FILE: {"rocprofiler-sdk-tool": [capture]},
        SCOPE_LEDGER_FILE: ledger,
        SCOPE_RECEIPT_FILE: bound.model_dump(mode="json"),
    }
    for name, payload in payloads.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(json.dumps(payload))
    return (
        intent,
        {name: sha256_file(tmp_path / name) for name in payloads},
        paths,
    )


def test_production_qualifies_implicit_dispatches(
    tmp_path, production_evidence
):
    from sol_execbench.core.bench.rocm_profiler.evaluate_scope import (
        qualify_evaluate_scope,
    )

    intent, hashes, paths = production_evidence
    result = qualify_evaluate_scope(
        tmp_path,
        intent,
        hashes,
        staging=tmp_path,
        collector=paths["collector_path"],
        returncode=0,
    )
    assert result.status == "qualified"
    assert result.implicit_dispatch_ids == [2]
    assert len(result.links) == 1
    assert not result.schedule_complete
    assert not result.performance_authority
    assert not result.adversarial_attestation
    assert result.copy_domain.qualified
    assert "default_stream_implicit_sync_not_inferred" in map(
        str, result.schedule.reason_codes
    )
    assert result.schedule_complete == result.schedule.schedule_complete


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "workload_uuid",
        "workload_sha256",
        "candidate_sha256",
        "definition_sha256",
        "device_uuid",
        "application_command",
        "collector_sha256",
    ],
)
def test_production_rejects_cross_identity(
    tmp_path, production_evidence, field
):
    from sol_execbench.core.bench.rocm_profiler.evaluate_scope import (
        qualify_evaluate_scope,
    )

    intent, hashes, paths = production_evidence
    replacement = {
        "run_id": "b" * 32,
        "workload_uuid": "another",
        "device_uuid": "GPU-0202020202020202",
        "application_command": ("other",),
    }.get(field, "f" * 64)
    changed = intent.model_copy(update={field: replacement})
    with pytest.raises(HIPScopeError):
        qualify_evaluate_scope(
            tmp_path,
            changed,
            hashes,
            staging=tmp_path,
            collector=paths["collector_path"],
            returncode=0,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "nonzero-exit",
        "missing-ledger",
        "staged-drift",
        "unsealed",
        "drop-dispatch",
        "graph",
        "wrong-process",
        "bad-correctness",
    ],
)
def test_production_rejects_missing_or_inconsistent_evidence(
    tmp_path, production_evidence, fault
):
    from sol_execbench.core.bench.rocm_profiler.evaluate_scope import (
        qualify_evaluate_scope,
    )
    from sol_execbench.core.bench.rocm_profiler.evaluate_scope_models import (
        SCOPE_LEDGER_FILE,
        SCOPE_RECEIPT_FILE,
        SCOPE_TRACE_FILE,
    )

    intent, hashes, paths = production_evidence
    if fault == "missing-ledger":
        hashes.pop(SCOPE_LEDGER_FILE)
    elif fault == "staged-drift":
        (tmp_path / "executor").write_text("changed")
    elif fault in ("unsealed", "wrong-process"):
        path = tmp_path / SCOPE_LEDGER_FILE
        data = json.loads(path.read_text())
        data["finalized" if fault == "unsealed" else "pid"] = (
            False if fault == "unsealed" else 99
        )
        path.write_text(json.dumps(data))
        hashes[str(path.relative_to(tmp_path))] = sha256_file(path)
    elif fault in ("drop-dispatch", "graph"):
        path = tmp_path / SCOPE_TRACE_FILE
        data = json.loads(path.read_text())
        rows = data["rocprofiler-sdk-tool"][0]["buffer_records"][
            "kernel_dispatch"
        ]
        if fault == "graph":
            rows[0]["graph_exec_id"] = 9
        else:
            rows.pop()
        path.write_text(json.dumps(data))
        hashes[str(path.relative_to(tmp_path))] = sha256_file(path)
    elif fault == "bad-correctness":
        path = tmp_path / SCOPE_RECEIPT_FILE
        data = json.loads(path.read_text())
        data["correct"] = False
        path.write_text(json.dumps(data))
        hashes[str(path.relative_to(tmp_path))] = sha256_file(path)
    with pytest.raises(HIPScopeError):
        qualify_evaluate_scope(
            tmp_path,
            intent,
            hashes,
            staging=tmp_path,
            collector=paths["collector_path"],
            returncode=1 if fault == "nonzero-exit" else 0,
        )
