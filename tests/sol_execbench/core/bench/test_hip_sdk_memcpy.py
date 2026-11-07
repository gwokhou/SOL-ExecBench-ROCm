# SPDX-License-Identifier: Apache-2.0
"""Engine-copy domain: channel agreement, lane binding, frozen vocabulary."""

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.rocm_profiler.hip_causality_json import SDKCapture
from sol_execbench.core.bench.rocm_profiler.hip_scope_artifacts import (
    HIPScopeError,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    HIPScopeReason,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_memcpy import (
    memcpy_copy_domain,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_models import (
    HIPSDKCopyDomain,
    HIPSDKCopyDomainReason,
    HIPSDKLedger,
    SDKBoundary,
)

API_NAMES = [
    "hipDeviceSynchronize",
    "hipMemcpyAsync",
    "hipLaunchKernel",
    "hipMemcpyDtoDAsync",
]
COPY_KIND = 2
DIRECTIONS = [
    "MEMORY_COPY_NONE",
    "MEMORY_COPY_HOST_TO_HOST",
    "MEMORY_COPY_HOST_TO_DEVICE",
    "MEMORY_COPY_DEVICE_TO_HOST",
    "MEMORY_COPY_DEVICE_TO_DEVICE",
]


def _tables():
    return {
        "buffer_records": [
            {"kind": "HIP_RUNTIME_API_EXT", "operations": API_NAMES},
            {
                "kind": "KERNEL_DISPATCH",
                "operations": ["KERNEL_DISPATCH_COMPLETE"],
            },
            {"kind": "MEMORY_COPY", "operations": DIRECTIONS},
        ]
    }


def _api_row(corr, name, ordinal, args, start=210):
    return {
        "kind": 0,
        "operation": API_NAMES.index(name),
        "correlation_id": {"internal": corr},
        "stream_id": {"handle": ordinal},
        "thread_id": 7,
        "start_timestamp": start,
        "end_timestamp": start + 10,
        "args": args,
        "retval": {"uint64_t_retval": 0},
    }


def _copy_row(corr, operation=4, start=300, bytes_=512):
    return {
        "kind": COPY_KIND,
        "operation": operation,
        "correlation_id": {"internal": corr},
        "thread_id": 7,
        "start_timestamp": start,
        "end_timestamp": start + 40,
        "src_agent_id": {"handle": 9},
        "dst_agent_id": {"handle": 9},
        "bytes": bytes_,
    }


def _dispatch_row(corr, queue, ordinal, dispatch_id):
    return {
        "kind": 1,
        "operation": 0,
        "correlation_id": {"internal": corr},
        "stream_id": {"handle": ordinal},
        "thread_id": 7,
        "start_timestamp": 400,
        "end_timestamp": 450,
        "dispatch_info": {
            "dispatch_id": dispatch_id,
            "kernel_id": dispatch_id,
            "agent_id": {"handle": 9},
            "queue_id": {"handle": queue},
        },
    }


def evidence():
    """One in-scope D2D engine copy on stream 2 with a co-observed kernel."""
    rows = [
        {"kind": "begin", "tid": 7, "timestamp": 200},
        {
            "kind": "api",
            "scope": 1,
            "tid": 7,
            "corr": 10,
            "name": "hipMemcpyAsync",
            "result": 0,
            "stream": 256,
        },
        {
            "kind": "memcpy",
            "tid": 7,
            "corr": 10,
            "operation": 4,
            "src_agent": 9,
            "dst_agent": 9,
            "bytes": 512,
            "start": 300,
            "end": 340,
        },
        {
            "kind": "enqueue",
            "scope": 1,
            "tid": 7,
            "corr": 11,
            "dispatch": 21,
            "agent": 9,
            "queue": 5,
            "kernel": 21,
        },
        {
            "kind": "api",
            "scope": 1,
            "tid": 7,
            "corr": 11,
            "name": "hipLaunchKernel",
            "result": 0,
        },
        {"kind": "end", "tid": 7, "timestamp": 1100},
        {"kind": "seal", "tid": 7, "timestamp": 1300},
    ]
    for index, row in enumerate(rows, 1):
        row["seq"] = index
    ledger = {
        "schema_version": "sol_execbench.hip_sdk_ledger.v2",
        "run_id": "a" * 32,
        "pid": 42,
        "records": rows,
        "row_count": len(rows),
        "drop_count": 0,
        "lossless": True,
        "finalized": True,
        "stop_ok": True,
        "flush_ok": True,
    }
    capture = {
        "metadata": {
            "pid": 42,
            "init_time": 1,
            "fini_time": 1400,
            "command": [],
        },
        "strings": _tables(),
        "buffer_records": {
            "hip_api": [
                _api_row(
                    10,
                    "hipMemcpyAsync",
                    2,
                    [{"name": "stream", "value": "0x100"}],
                ),
                _api_row(11, "hipLaunchKernel", 2, [], start=260),
            ],
            "kernel_dispatch": [_dispatch_row(11, 5, 2, 21)],
            "hip_event": [],
            "memory_copy": [_copy_row(10)],
        },
    }
    return ledger, capture


def domain(ledger, capture):
    model = HIPSDKLedger.model_validate(ledger)
    begin = next(
        r
        for r in model.records
        if isinstance(r, SDKBoundary) and r.kind == "begin"
    )
    end = next(
        r
        for r in model.records
        if isinstance(r, SDKBoundary) and r.kind == "end"
    )
    return memcpy_copy_domain(
        model, SDKCapture.model_validate(capture), begin, end
    )


def test_engine_copy_binds_lane_and_qualifies():
    result = domain(*evidence())
    assert result.qualified
    assert not result.reason_codes
    assert len(result.copies) == 1
    copy = result.copies[0]
    assert copy.in_scope
    assert copy.queue_id == 5
    assert copy.stream_ordinal == 2
    assert copy.stream_handle == 256
    assert copy.direction == "MEMORY_COPY_DEVICE_TO_DEVICE"
    assert (
        HIPSDKCopyDomain.model_validate_json(result.model_dump_json()) == result
    )


def test_zero_engine_copies_qualify_vacuously():
    ledger, capture = evidence()
    capture["buffer_records"]["memory_copy"] = []
    rows = [r for r in ledger["records"] if r["kind"] != "memcpy"]
    for index, row in enumerate(rows, 1):
        row["seq"] = index
    ledger["records"] = rows
    ledger["row_count"] = len(rows)
    result = domain(ledger, capture)
    assert result.qualified
    assert not result.copies


def test_chunked_large_copy_shares_correlation_and_qualifies():
    ledger, capture = evidence()
    capture["buffer_records"]["hip_api"][0]["args"] = [
        {"name": "stream", "value": "0x100"},
        {"name": "sizeBytes", "value": "1024"},
    ]
    rows = ledger["records"]
    memcpy_row = next(r for r in rows if r["kind"] == "memcpy")
    memcpy_row["bytes"] = 768
    capture["buffer_records"]["memory_copy"][0]["bytes"] = 768
    chunk = {
        "kind": "memcpy",
        "tid": 7,
        "corr": 10,
        "operation": 4,
        "src_agent": 9,
        "dst_agent": 9,
        "bytes": 256,
        "start": 341,
        "end": 350,
    }
    rows.insert(rows.index(memcpy_row) + 1, chunk)
    for index, row in enumerate(rows, 1):
        row["seq"] = index
    ledger["records"] = rows
    ledger["row_count"] = len(rows)
    capture["buffer_records"]["memory_copy"].append(
        {
            "kind": 2,
            "operation": 4,
            "correlation_id": {"internal": 10},
            "thread_id": 7,
            "start_timestamp": 341,
            "end_timestamp": 350,
            "src_agent_id": {"handle": 9},
            "dst_agent_id": {"handle": 9},
            "bytes": 256,
        }
    )
    result = domain(ledger, capture)
    assert result.qualified
    assert not result.reason_codes
    assert [c.bytes for c in result.copies] == [768, 256]
    assert all(c.correlation_id == 10 for c in result.copies)


def test_chunk_bytes_must_sum_to_host_size_bytes():
    ledger, capture = evidence()
    capture["buffer_records"]["hip_api"][0]["args"] = [
        {"name": "stream", "value": "0x100"},
        {"name": "sizeBytes", "value": "1024"},
    ]
    rows = ledger["records"]
    memcpy_row = next(r for r in rows if r["kind"] == "memcpy")
    chunk = {
        "kind": "memcpy",
        "tid": 7,
        "corr": 10,
        "operation": 4,
        "src_agent": 9,
        "dst_agent": 9,
        "bytes": 256,
        "start": 341,
        "end": 350,
    }
    rows.insert(rows.index(memcpy_row) + 1, chunk)
    for index, row in enumerate(rows, 1):
        row["seq"] = index
    ledger["records"] = rows
    ledger["row_count"] = len(rows)
    capture["buffer_records"]["memory_copy"].append(
        {
            "kind": 2,
            "operation": 4,
            "correlation_id": {"internal": 10},
            "thread_id": 7,
            "start_timestamp": 341,
            "end_timestamp": 350,
            "src_agent_id": {"handle": 9},
            "dst_agent_id": {"handle": 9},
            "bytes": 256,
        }
    )
    with pytest.raises(HIPScopeError) as error:
        domain(ledger, capture)
    assert error.value.reason is HIPScopeReason.COPY


def test_outside_copy_is_retained_without_reasons():
    ledger, capture = evidence()
    rows = ledger["records"]
    api = next(r for r in rows if r.get("name") == "hipMemcpyAsync")
    api["scope"] = 0
    rows.insert(0, rows.pop(rows.index(api)))
    for index, row in enumerate(rows, 1):
        row["seq"] = index
    result = domain(ledger, capture)
    assert result.qualified
    assert result.outside_correlation_ids == (10,)
    assert result.copies[0].queue_id is None
    assert not result.copies[0].in_scope


@pytest.mark.parametrize(
    "fault",
    [
        "ledger_row_deleted",
        "export_row_deleted",
        "corr_rebound",
        "bytes_changed",
        "timestamps_changed",
        "direction_changed",
        "outside_lifetime",
        "stream_argument_conflict",
    ],
)
def test_copy_channel_mismatch_rejects(fault):
    ledger, capture = evidence()
    rows = ledger["records"]
    if fault == "ledger_row_deleted":
        rows = [r for r in rows if r["kind"] != "memcpy"]
    elif fault == "export_row_deleted":
        capture["buffer_records"]["memory_copy"] = []
    elif fault == "corr_rebound":
        rows = [dict(r, corr=99) if r["kind"] == "memcpy" else r for r in rows]
    elif fault == "bytes_changed":
        capture["buffer_records"]["memory_copy"][0]["bytes"] = 513
    elif fault == "timestamps_changed":
        capture["buffer_records"]["memory_copy"][0]["end_timestamp"] = 341
    elif fault == "direction_changed":
        capture["buffer_records"]["memory_copy"][0]["operation"] = 3
    elif fault == "outside_lifetime":
        capture["buffer_records"]["memory_copy"][0]["start_timestamp"] = 9999
    elif fault == "stream_argument_conflict":
        capture["buffer_records"]["hip_api"][0]["args"] = [
            {"name": "stream", "value": "0x200"}
        ]
    for index, row in enumerate(rows, 1):
        row["seq"] = index
    ledger["records"] = rows
    ledger["row_count"] = len(rows)
    with pytest.raises(HIPScopeError) as error:
        domain(ledger, capture)
    assert error.value.reason is HIPScopeReason.COPY


def test_unsupported_api_and_direction_stay_reason_codes():
    ledger, capture = evidence()
    rows = ledger["records"]
    api = next(r for r in rows if r.get("name") == "hipMemcpyAsync")
    api["name"] = "hipMemcpyDtoDAsync"
    capture["buffer_records"]["hip_api"][0]["operation"] = API_NAMES.index(
        "hipMemcpyDtoDAsync"
    )
    result = domain(ledger, capture)
    assert not result.qualified
    assert HIPSDKCopyDomainReason.UNSUPPORTED_API in result.reason_codes
    assert result.copies[0].queue_id is None


def test_unsupported_direction_rejects_domain():
    ledger, capture = evidence()
    rows = ledger["records"]
    copy_row = next(r for r in rows if r["kind"] == "memcpy")
    copy_row["operation"] = 1
    capture["buffer_records"]["memory_copy"][0]["operation"] = 1
    result = domain(ledger, capture)
    assert HIPSDKCopyDomainReason.UNSUPPORTED_DIRECTION in result.reason_codes
    assert not result.qualified


def test_queue_unresolved_without_co_observation():
    ledger, capture = evidence()
    capture["buffer_records"]["kernel_dispatch"] = []
    result = domain(ledger, capture)
    assert HIPSDKCopyDomainReason.QUEUE_UNRESOLVED in result.reason_codes
    assert not result.qualified
    assert result.copies[0].queue_id is None


def test_stream_conflict_rejects_domain():
    ledger, capture = evidence()
    capture["buffer_records"]["kernel_dispatch"].append(
        _dispatch_row(12, 6, 2, 22)
    )
    result = domain(ledger, capture)
    assert HIPSDKCopyDomainReason.STREAM_CONFLICT in result.reason_codes
    assert not result.qualified


def test_default_stream_copy_never_binds_a_lane():
    ledger, capture = evidence()
    rows = ledger["records"]
    api = next(r for r in rows if r.get("name") == "hipMemcpyAsync")
    api["stream"] = 0
    capture["buffer_records"]["hip_api"][0]["stream_id"]["handle"] = 0
    capture["buffer_records"]["hip_api"][0]["args"] = [
        {"name": "stream", "value": "0"}
    ]
    result = domain(ledger, capture)
    assert HIPSDKCopyDomainReason.DEFAULT_STREAM in result.reason_codes
    assert result.copies[0].queue_id is None
    assert result.copies[0].in_scope
    assert not result.qualified


def test_qualified_domain_rejects_unresolved_lanes_at_model_level():
    with pytest.raises(ValidationError):
        HIPSDKCopyDomain(
            copies=[
                {
                    "correlation_id": 1,
                    "direction": "MEMORY_COPY_DEVICE_TO_DEVICE",
                    "bytes": 8,
                    "start_ns": 1,
                    "end_ns": 2,
                    "stream_handle": 0,
                    "stream_ordinal": 2,
                    "queue_id": None,
                    "in_scope": True,
                }
            ],
            qualified=True,
        )
