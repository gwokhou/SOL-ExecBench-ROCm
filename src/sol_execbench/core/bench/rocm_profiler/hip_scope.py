# SPDX-License-Identifier: Apache-2.0
"""Verify complete controlled-call coverage and bind every observed dispatch."""

from itertools import pairwise
from pathlib import Path

from sol_execbench.core.bench.rocm_profiler.hip_causality import (
    correlate_hip_events,
)
from sol_execbench.core.bench.rocm_profiler.hip_causality_json import (
    SDKCapture,
    causal_trace,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_artifacts import (
    HIPScopeCaptureView,
    HIPScopeError,
    read_bound_json,
    read_scope_capture,
    scope_agent,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    HIPBoundDispatch,
    HIPJournalAPI,
    HIPScopeExpectation,
    HIPScopeReason,
    HIPTraceJournal,
    HIPTraceScopeQualification,
)
from sol_execbench.core.integrity.checksums import sha256_file


def qualify_hip_trace_scope(
    trace_path: Path,
    journal_path: Path,
    candidate_path: Path,
    expected: HIPScopeExpectation,
    *,
    trace_sha256: str,
    journal_sha256: str,
) -> HIPTraceScopeQualification:
    """Check evidence against a trusted controller's audited-executor scope.

    This API cannot audit arbitrary candidate code. The caller owns that trust
    decision and must retain the pre-run identity and collection return code.
    """
    try:
        capture, view = read_scope_capture(
            read_bound_json(trace_path, trace_sha256)
        )
        journal = HIPTraceJournal.model_validate_json(
            read_bound_json(journal_path, journal_sha256)
        )
        if sha256_file(candidate_path) != expected.candidate_sha256:
            raise HIPScopeError(HIPScopeReason.IDENTITY)
        agent = scope_agent(capture, view, journal, expected)
        trace = causal_trace(capture, f"{trace_path.name}:0", trace_sha256)
        correlations = _verify_journal(capture, journal)
        dispatches = _verify_dispatches(
            capture, view, journal, correlations, agent
        )
        _verify_barriers(capture, correlations, journal, agent)
        _verify_fences(capture, correlations, journal)
        causality = correlate_hip_events(trace)
        if causality.reason_codes or causality.unresolved_waits:
            raise HIPScopeError(HIPScopeReason.CAUSALITY)
        return HIPTraceScopeQualification(
            scope=expected,
            trace_sha256=trace_sha256,
            journal_sha256=journal_sha256,
            process_id=journal.process_id,
            dispatches=dispatches,
            links=causality.links,
            journal_call_count=len(journal.calls),
        )
    except HIPScopeError:
        raise
    except OSError, ValueError, IndexError, KeyError, RecursionError:
        raise HIPScopeError(HIPScopeReason.INVALID) from None


def _verify_journal(capture: SDKCapture, journal: HIPTraceJournal) -> list[int]:
    calls = sorted(
        capture.buffer_records.hip_api, key=lambda call: call.start_timestamp
    )
    if (
        len({call.correlation_id.internal for call in calls}) != len(calls)
        or any(call.thread_id != journal.thread_id for call in calls)
        or any(a.end_timestamp >= b.start_timestamp for a, b in pairwise(calls))
    ):
        raise HIPScopeError(HIPScopeReason.JOURNAL)
    selected = []
    for call in calls:
        name = capture.operation(call, "HIP_RUNTIME_API_EXT")
        if (
            "Graph" in name
            or "Capture" in name
            or "Ipc" in name
            or name.endswith("_spt")
        ):
            raise HIPScopeError(HIPScopeReason.JOURNAL)
        if name in HIPJournalAPI:
            selected.append(call)
    if len(selected) != len(journal.calls):
        raise HIPScopeError(HIPScopeReason.JOURNAL)
    for call, witness in zip(selected, journal.calls, strict=True):
        args = {arg.name: arg.value for arg in call.args}
        if (
            capture.operation(call, "HIP_RUNTIME_API_EXT") != witness.name
            or call.retval.uint64_t_retval != 0
            or witness.result != 0
            or int(args.get("flags", "0"), 0) != 0
            or int(args.get("stream", "0"), 0) != witness.stream_handle
            or int(args.get("event", "0"), 0) != witness.event_handle
        ):
            raise HIPScopeError(HIPScopeReason.JOURNAL)
    return [call.correlation_id.internal for call in selected]


def _verify_dispatches(
    capture: SDKCapture,
    view: HIPScopeCaptureView,
    journal: HIPTraceJournal,
    correlations: list[int],
    agent: int,
) -> list[HIPBoundDispatch]:
    launches = {
        identifier: call
        for identifier, call in zip(correlations, journal.calls, strict=True)
        if call.name == HIPJournalAPI.LAUNCH
    }
    records = capture.buffer_records.kernel_dispatch
    if (
        len(records) != len(launches)
        or {item.correlation_id.internal for item in records} != set(launches)
        or len({item.dispatch_info.dispatch_id for item in records})
        != len(records)
        or len({symbol.kernel_id for symbol in view.kernel_symbols})
        != len(view.kernel_symbols)
    ):
        raise HIPScopeError(HIPScopeReason.DISPATCH)
    symbols = {
        item.kernel_id: item.kernel_name.removesuffix(".kd")
        for item in view.kernel_symbols
    }
    hosts = {
        call.correlation_id.internal: call
        for call in capture.buffer_records.hip_api
    }
    bound = []
    for item in records:
        info = item.dispatch_info
        identifier = item.correlation_id.internal
        name = symbols.get(info.kernel_id)
        if (
            name is None
            or name != launches[identifier].kernel_name
            or info.agent_id.handle != agent
            or item.thread_id != journal.thread_id
            or item.stream_id.handle == 0
            or item.stream_id.handle != hosts[identifier].stream_id.handle
        ):
            raise HIPScopeError(HIPScopeReason.DISPATCH)
        bound.append(
            HIPBoundDispatch(
                dispatch_id=info.dispatch_id,
                correlation_id=identifier,
                kernel_name=name,
                agent_id=agent,
                queue_id=info.queue_id.handle,
                stream_id=item.stream_id.handle,
            )
        )
    return sorted(bound, key=lambda item: item.dispatch_id)


def _verify_barriers(
    capture: SDKCapture,
    correlations: list[int],
    journal: HIPTraceJournal,
    agent: int,
) -> None:
    expected = {
        identifier
        for identifier, call in zip(correlations, journal.calls, strict=True)
        if call.name in {HIPJournalAPI.RECORD, HIPJournalAPI.WAIT}
    }
    barriers = capture.buffer_records.hip_event
    if (
        len(barriers) != len(expected)
        or {item.correlation_id.internal for item in barriers} != expected
        or any(item.agent_id.handle != agent for item in barriers)
    ):
        raise HIPScopeError(HIPScopeReason.BARRIER)
    hosts = {
        call.correlation_id.internal: call
        for call in capture.buffer_records.hip_api
    }
    for item in barriers:
        host = hosts[item.correlation_id.internal]
        args = {arg.name: arg.value for arg in host.args}
        operation = (
            "HIP_EVENT_RECORD"
            if capture.operation(host, "HIP_RUNTIME_API_EXT")
            == HIPJournalAPI.RECORD
            else "HIP_EVENT_WAIT"
        )
        if (
            capture.operation(item, "HIP_EVENT") != operation
            or int(args["event"], 0) != item.hip_event_handle
            or item.stream_id.handle != host.stream_id.handle
            or item.thread_id != journal.thread_id
        ):
            raise HIPScopeError(HIPScopeReason.BARRIER)


def _verify_fences(
    capture: SDKCapture, correlations: list[int], journal: HIPTraceJournal
) -> None:
    syncs = [
        identifier
        for identifier, call in zip(correlations, journal.calls, strict=True)
        if call.name == HIPJournalAPI.SYNC
    ]
    hosts = {
        call.correlation_id.internal: call
        for call in capture.buffer_records.hip_api
    }
    if len(syncs) < 2:
        raise HIPScopeError(HIPScopeReason.FENCE)
    begin, end = hosts[syncs[0]], hosts[syncs[-1]]
    if begin.end_timestamp >= end.start_timestamp:
        raise HIPScopeError(HIPScopeReason.FENCE)
    for item in (
        *capture.buffer_records.kernel_dispatch,
        *capture.buffer_records.hip_event,
    ):
        host = hosts[item.correlation_id.internal]
        if (
            not begin.end_timestamp
            < host.start_timestamp
            < host.end_timestamp
            < end.start_timestamp
        ):
            raise HIPScopeError(HIPScopeReason.FENCE)
    # HIP event record timestamps may extend beyond device synchronization.
    # Completion containment is checked on hardware dispatch records only.
    for dispatch in capture.buffer_records.kernel_dispatch:
        if (
            not begin.end_timestamp
            <= dispatch.start_timestamp
            <= dispatch.end_timestamp
            <= end.end_timestamp
        ):
            raise HIPScopeError(HIPScopeReason.FENCE)
