# SPDX-License-Identifier: Apache-2.0
"""Acceptance cases for host-order links and conservative rejection."""

import copy
import json
from dataclasses import replace

import pytest

from sol_execbench.core.bench.profile_summary import (
    ProfileSummarySidecar,
    build_profile_summary_sidecar,
)
from sol_execbench.core.bench.profile_summary.artifacts import (
    structured_profile_evidence,
)
from sol_execbench.core.bench.rocm_profiler.hip_causality_artifacts import (
    read_hip_event_causality,
)
from sol_execbench.core.bench.rocm_profiler.hip_causality_models import (
    HIPEventCausality,
    HIPLinkReason,
)
from sol_execbench.core.bench.rocm_profiler.models import (
    Rocprofv3ArtifactKind,
    Rocprofv3ProfileArtifact,
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
)

API_NAMES = [
    "hipLaunchKernel",
    "hipEventRecord",
    "hipStreamWaitEvent",
    "hipEventDestroy",
]


def activity(kind, operation, correlation, stream, start, end):
    return {
        "kind": kind,
        "operation": operation,
        "thread_id": 7,
        "correlation_id": {"internal": correlation},
        "stream_id": {"handle": stream},
        "start_timestamp": start,
        "end_timestamp": end,
    }


def host(name, correlation, stream, start):
    item = activity(
        0, API_NAMES.index(name), correlation, stream, start, start + 1
    )
    item.update(args=[], retval={"uint64_t_retval": 0})
    if name != "hipLaunchKernel":
        item["args"] = [{"name": "event", "value": "0xabc"}]
    if name == "hipStreamWaitEvent":
        item["args"].append({"name": "flags", "value": "0"})
    return item


def barrier(record, correlation, stream, start, end):
    item = activity(1, int(not record), correlation, stream, start, end)
    item.update(
        agent_id={"handle": 9},
        queue_id={"handle": stream * 2},
        source_queue_id={"handle": 2},
        hip_event_handle=0xABC,
    )
    return item


def dispatch(identifier, correlation, stream):
    item = activity(2, 0, correlation, stream, 300, 310)
    item["dispatch_info"] = {
        "dispatch_id": identifier,
        "agent_id": {"handle": 9},
        "queue_id": {"handle": stream * 2},
    }
    return item


@pytest.fixture
def capture():
    return {
        "metadata": {"pid": 42, "init_time": 1, "fini_time": 1000},
        "strings": {
            "buffer_records": [
                {"kind": "HIP_RUNTIME_API_EXT", "operations": API_NAMES},
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
            "hip_api": [
                host("hipLaunchKernel", 1, 1, 10),
                host("hipEventRecord", 2, 1, 20),
                host("hipStreamWaitEvent", 3, 2, 30),
                host("hipLaunchKernel", 4, 2, 40),
                host("hipStreamWaitEvent", 5, 3, 50),
                host("hipLaunchKernel", 6, 3, 60),
                host("hipLaunchKernel", 10, 4, 90),
                host("hipLaunchKernel", 7, 1, 180),
                host("hipEventRecord", 8, 1, 200),
                host("hipStreamWaitEvent", 9, 2, 210),
                host("hipLaunchKernel", 11, 2, 220),
            ],
            "hip_event": [
                barrier(True, 2, 1, 25, 100),
                barrier(False, 3, 2, 35, 110),
                barrier(False, 5, 3, 55, 120),
                barrier(True, 8, 1, 205, 250),
                barrier(False, 9, 2, 215, 260),
            ],
            "kernel_dispatch": [
                dispatch(11, 1, 1),
                dispatch(12, 4, 2),
                dispatch(13, 6, 3),
                dispatch(14, 7, 1),
                dispatch(15, 11, 2),
                dispatch(16, 10, 4),
            ],
        },
    }


def read(tmp_path, *captures):
    path = tmp_path / "capture.json"
    path.write_text(json.dumps({"rocprofiler-sdk-tool": captures}))
    return read_hip_event_causality(path)


def edges(result):
    return [
        (
            x.predecessor_dispatch_id,
            x.successor_dispatch_id,
            x.record_correlation_id,
        )
        for x in result.links
    ]


def test_fanout_rerecord_and_independent_stream(capture, tmp_path):
    results, warnings = read(tmp_path, capture)
    assert not warnings
    assert edges(results[0]) == [(11, 12, 2), (11, 13, 2), (14, 15, 8)]
    assert not results[0].unresolved_waits
    assert results[0].status == "partial"
    assert (
        "trace_completeness_not_independently_proven" in results[0].limitations
    )
    assert "dispatch_scope_not_bound_to_workload" in results[0].limitations


def test_output_order_and_gpu_completion_order_do_not_choose_generation(
    capture, tmp_path
):
    capture["buffer_records"]["hip_event"][1]["end_timestamp"] = 90
    for rows in capture["buffer_records"].values():
        rows.reverse()
    results, warnings = read(tmp_path, capture)
    assert not warnings
    assert edges(results[0]) == [(11, 12, 2), (11, 13, 2), (14, 15, 8)]


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("missing_record_host", HIPLinkReason.MISSING_HOST),
        ("missing_record_barrier", HIPLinkReason.MISSING_HOST),
        ("missing_wait_barrier", HIPLinkReason.MISSING_HOST),
        ("failed_wait", HIPLinkReason.HOST_MISMATCH),
        ("nondefault_wait_flags", HIPLinkReason.HOST_MISMATCH),
        ("mismatched_handle", HIPLinkReason.HOST_MISMATCH),
        ("mismatched_device", HIPLinkReason.DEVICE),
        ("mismatched_source_queue", HIPLinkReason.DEVICE),
        ("different_host_thread", HIPLinkReason.AMBIGUOUS_ORDER),
        ("overlapping_host_calls", HIPLinkReason.AMBIGUOUS_ORDER),
        ("default_stream", HIPLinkReason.HOST_MISMATCH),
        ("destroy_before_wait", HIPLinkReason.MISSING_RECORD),
        ("missing_launch_host", HIPLinkReason.MISSING_HOST),
        ("missing_producer", HIPLinkReason.MISSING_DISPATCH),
    ],
)
def test_incomplete_or_ambiguous_waits_never_get_an_edge(
    capture, tmp_path, mutation, reason
):
    records = capture["buffer_records"]
    calls, barriers, kernels = (
        records[k] for k in ("hip_api", "hip_event", "kernel_dispatch")
    )
    if mutation == "missing_record_host":
        calls.pop(1)
    elif mutation == "missing_record_barrier":
        barriers.pop(0)
    elif mutation == "missing_wait_barrier":
        barriers.pop(1)
    elif mutation == "failed_wait":
        calls[2]["retval"]["uint64_t_retval"] = 1
    elif mutation == "nondefault_wait_flags":
        calls[2]["args"][1]["value"] = "1"
    elif mutation == "mismatched_handle":
        barriers[1]["hip_event_handle"] += 1
    elif mutation == "mismatched_device":
        barriers[1]["agent_id"]["handle"] += 1
    elif mutation == "mismatched_source_queue":
        barriers[1]["source_queue_id"]["handle"] += 1
    elif mutation == "different_host_thread":
        calls[2]["thread_id"] = barriers[1]["thread_id"] = 8
    elif mutation == "overlapping_host_calls":
        calls[1]["end_timestamp"] = 31
    elif mutation == "default_stream":
        calls[2]["stream_id"]["handle"] = barriers[1]["stream_id"]["handle"] = 0
    elif mutation == "destroy_before_wait":
        calls.append(host("hipEventDestroy", 22, 0, 25))
    elif mutation == "missing_launch_host":
        calls.pop(0)
    else:
        kernels.pop(0)
    results, warnings = read(tmp_path, capture)
    assert not warnings
    assert not any(x.wait_correlation_id == 3 for x in results[0].links)
    assert any(
        x.correlation_id == 3 and x.reason == reason
        for x in results[0].unresolved_waits
    )


def test_record_reuse_while_gpu_record_is_pending_is_rejected(
    capture, tmp_path
):
    capture["buffer_records"]["hip_event"][0]["end_timestamp"] = 201
    results, _ = read(tmp_path, capture)
    assert edges(results[0]) == [(11, 12, 2), (11, 13, 2)]
    assert (
        results[0].unresolved_waits[0].reason == HIPLinkReason.IN_FLIGHT_RECORD
    )


@pytest.mark.parametrize("domain", ["hip_api", "hip_event", "kernel_dispatch"])
def test_duplicate_identity_rejects_capture(capture, tmp_path, domain):
    capture["buffer_records"][domain].append(
        copy.deepcopy(capture["buffer_records"][domain][0])
    )
    results, _ = read(tmp_path, capture)
    assert not results[0].links
    assert results[0].reason_codes == [HIPLinkReason.DUPLICATE]


def test_processes_and_captures_never_supply_each_others_missing_calls(
    capture, tmp_path
):
    other = copy.deepcopy(capture)
    other["metadata"]["pid"] = 43
    capture["buffer_records"]["hip_api"].pop(1)
    results, _ = read(tmp_path, capture, other)
    assert not any(x.wait_correlation_id == 3 for x in results[0].links)
    assert len(results[1].links) == 3
    assert results[0].capture != results[1].capture


@pytest.mark.parametrize(
    "mutation",
    ["negative_id", "unfinished", "outside_lifetime", "graph", "row_limit"],
)
def test_unsupported_or_malformed_capture_is_not_partially_joined(
    capture, tmp_path, mutation
):
    if mutation == "negative_id":
        capture["buffer_records"]["hip_api"][0]["correlation_id"][
            "internal"
        ] = -1
    elif mutation == "unfinished":
        capture["metadata"]["fini_time"] = 0
    elif mutation == "outside_lifetime":
        capture["metadata"]["fini_time"] = 200
    elif mutation == "graph":
        capture["buffer_records"]["kernel_dispatch"][0]["graph_exec_id"] = 1
    else:
        capture["buffer_records"]["hip_api"] *= 1000
    results, warnings = read(tmp_path, capture)
    assert not results
    assert warnings


def test_unrecorded_wait_without_a_gpu_barrier_is_reported(capture, tmp_path):
    capture["buffer_records"]["hip_event"] = []
    results, _ = read(tmp_path, capture)
    assert not results[0].links
    assert len(results[0].unresolved_waits) == 3


def test_summary_exposes_links_only_for_successful_profile(capture, tmp_path):
    read(tmp_path, capture)
    profile = Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.SUCCESS,
        command=("rocprofv3",),
        returncode=0,
        output_directory=tmp_path,
        output_file="capture",
        artifacts=(
            Rocprofv3ProfileArtifact(
                path=tmp_path / "capture.json",
                kind=Rocprofv3ArtifactKind.METADATA_JSON,
                size_bytes=(tmp_path / "capture.json").stat().st_size,
            ),
        ),
    )
    evidence = structured_profile_evidence(profile)
    assert len(evidence.hip_event_causality[0].links) == 3
    sidecar = build_profile_summary_sidecar(profile_result=profile)
    assert sidecar.summary.hip_event_causality == evidence.hip_event_causality
    assert (
        ProfileSummarySidecar.model_validate(
            json.loads(sidecar.model_dump_json())
        )
        == sidecar
    )
    failed = replace(
        profile,
        status=Rocprofv3ProfileStatus.FAILED,
        failed_reason="collection failed",
    )
    assert not structured_profile_evidence(failed).hip_event_causality


def test_missing_rerecord_host_never_falls_back_to_old_generation(
    capture, tmp_path
):
    capture["buffer_records"]["hip_api"].pop(8)
    results, _ = read(tmp_path, capture)
    assert not results[0].links
    assert results[0].reason_codes == [HIPLinkReason.MISSING_HOST]


def test_missing_wait_flags_rejects_artifact(capture, tmp_path):
    capture["buffer_records"]["hip_api"][2]["args"].pop()
    results, warnings = read(tmp_path, capture)
    assert not results
    assert warnings


@pytest.mark.parametrize(
    "missing_host", [False, True], ids=["linked", "unresolved"]
)
def test_public_payload_roundtrips_through_decoded_json(
    capture, tmp_path, missing_host
):
    if missing_host:
        capture["buffer_records"]["hip_api"].pop(1)
    results, _ = read(tmp_path, capture)
    payload = json.loads(results[0].model_dump_json())
    assert HIPEventCausality.model_validate(payload) == results[0]


@pytest.mark.parametrize("name", ["hipStreamDestroy", "hipDeviceReset"])
def test_stream_lifetime_changes_before_wait_reject_capture(
    capture, tmp_path, name
):
    table = capture["strings"]["buffer_records"][0]["operations"] = list(
        API_NAMES
    )
    table.append(name)
    call = host("hipLaunchKernel", 90, 0, 27)
    call["operation"] = len(table) - 1
    capture["buffer_records"]["hip_api"].append(call)
    results, warnings = read(tmp_path, capture)
    assert not results
    assert warnings == ["capture.json: hip_causality_unsupported_capture"]


def test_destroy_without_handle_rejects_capture(capture, tmp_path):
    call = host("hipEventDestroy", 90, 0, 27)
    call["args"] = []
    capture["buffer_records"]["hip_api"].append(call)
    results, warnings = read(tmp_path, capture)
    assert not results
    assert warnings


@pytest.mark.parametrize(
    "successful", [True, False], ids=["completed", "failed"]
)
def test_explicit_event_completion_resolves_late_barrier_observation(
    capture, tmp_path, successful
):
    capture["buffer_records"]["hip_event"][0]["end_timestamp"] = 201
    table = capture["strings"]["buffer_records"][0]["operations"] = [
        *API_NAMES,
        "hipEventSynchronize",
    ]
    sync = host("hipEventDestroy", 91, 0, 170)
    sync["operation"] = len(table) - 1
    sync["retval"]["uint64_t_retval"] = int(not successful)
    capture["buffer_records"]["hip_api"].append(sync)
    results, warnings = read(tmp_path, capture)
    assert not warnings
    assert len(results[0].links) == (3 if successful else 2)
