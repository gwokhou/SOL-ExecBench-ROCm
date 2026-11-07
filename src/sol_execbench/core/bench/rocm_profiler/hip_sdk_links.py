# SPDX-License-Identifier: Apache-2.0
"""Event links from a qualified SDK enqueue ledger, including engine copies."""

from collections.abc import Sequence
from typing import NamedTuple

from sol_execbench.core.bench.rocm_profiler.hip_causality_json import (
    SDKCapture,
    normalize_host_call,
)
from sol_execbench.core.bench.rocm_profiler.hip_causality_models import (
    RECORD_APIS,
    WAIT_API,
    HIPHostCall,
)
from sol_execbench.core.bench.rocm_profiler.hip_events import HIPEventOperation
from sol_execbench.core.bench.rocm_profiler.hip_scope_artifacts import (
    HIPScopeError,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    HIPScopeReason,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_models import (
    SDKAPI,
    HIPSDKEngineCopy,
    HIPSDKEventLink,
    HIPSDKLedger,
    SDKEnqueue,
    SDKEventEnqueue,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_validation import (
    require_scope,
)


def sdk_event_links(
    ledger: HIPSDKLedger,
    capture: SDKCapture,
    enqueues: list[SDKEnqueue],
    copies: Sequence[HIPSDKEngineCopy] = (),
) -> list[HIPSDKEventLink]:
    """Support one record generation per event; ambiguity stays unqualified."""
    apis = {
        r.corr: r
        for r in ledger.records
        if isinstance(r, SDKAPI) and r.scope == 1
    }
    events = [
        r
        for r in ledger.records
        if isinstance(r, SDKEventEnqueue) and r.scope == 1
    ]
    expected = {
        r.corr for r in apis.values() if r.name in RECORD_APIS | {WAIT_API}
    }
    require_scope(expected == {r.corr for r in events}, HIPScopeReason.BARRIER)
    raw = {
        r.correlation_id.internal: r for r in capture.buffer_records.hip_event
    }
    host = {
        r.correlation_id.internal: normalize_host_call(capture, r)
        for r in capture.buffer_records.hip_api
    }
    for event in events:
        call = apis[event.corr]
        operation = capture.operation(raw[event.corr], "HIP_EVENT")
        require_scope(
            call.result == 0
            and host[event.corr].event_handle == event.event
            and host[event.corr].flags == 0
            and host[event.corr].thread_id == event.tid
            and host[event.corr].stream_id == raw[event.corr].stream_id.handle
            and event.seq < call.seq
            and operation
            == (
                HIPEventOperation.WAIT
                if call.name == WAIT_API
                else HIPEventOperation.RECORD
            ),
            HIPScopeReason.BARRIER,
        )
    return _event_links(events, apis, host, enqueues, copies)


def _link(
    record: SDKEventEnqueue,
    wait: SDKEventEnqueue,
    enqueues: list[SDKEnqueue],
    copies: Sequence[HIPSDKEngineCopy],
) -> HIPSDKEventLink:
    producer = _endpoint(
        record.queue,
        [r for r in enqueues if r.scope == 1 and r.seq < record.seq],
        [
            c
            for c in copies
            if c.queue_id is not None and c.submission_seq < record.seq
        ],
        last=True,
    )
    consumer = _endpoint(
        wait.queue,
        [r for r in enqueues if r.scope == 1 and r.seq > wait.seq],
        [
            c
            for c in copies
            if c.queue_id is not None and c.submission_seq > wait.seq
        ],
        last=False,
    )
    require_scope(
        producer is not None and consumer is not None, HIPScopeReason.CAUSALITY
    )
    if producer is None or consumer is None:
        raise HIPScopeError(HIPScopeReason.CAUSALITY)
    return HIPSDKEventLink(
        predecessor_dispatch_id=producer.dispatch,
        successor_dispatch_id=consumer.dispatch,
        predecessor_copy_correlation_id=producer.copy_corr,
        successor_copy_correlation_id=consumer.copy_corr,
        record_correlation_id=record.corr,
        wait_correlation_id=wait.corr,
        event_handle=wait.event,
    )


class _Submission(NamedTuple):
    """One ordered submission: a kernel dispatch or a lane-bound engine copy."""

    seq: int
    queue: int
    dispatch: int | None
    copy_corr: int | None


def _endpoint(
    queue: int,
    enqueues: list[SDKEnqueue],
    copies: Sequence[HIPSDKEngineCopy],
    *,
    last: bool,
) -> _Submission | None:
    """Pick the adjacent same-queue submission without timestamp ordering."""
    submissions = [
        _Submission(r.seq, r.queue, r.dispatch, None)
        for r in enqueues
        if r.queue == queue
    ] + [
        _Submission(c.submission_seq, queue, None, c.correlation_id)
        for c in copies
        if c.queue_id == queue
    ]
    if not submissions:
        return None
    key = max if last else min
    return key(submissions, key=lambda submission: submission.seq)


def _event_links(
    events: list[SDKEventEnqueue],
    apis: dict[int, SDKAPI],
    host: dict[int, HIPHostCall],
    enqueues: list[SDKEnqueue],
    copies: Sequence[HIPSDKEngineCopy],
) -> list[HIPSDKEventLink]:
    """Resolve waits against the single qualified event generation."""
    records = [r for r in events if apis[r.corr].name in RECORD_APIS]
    require_scope(
        len({r.event for r in records}) == len(records),
        HIPScopeReason.CAUSALITY,
    )
    by_handle = {r.event: r for r in records}
    links = []
    for wait in (r for r in events if apis[r.corr].name == WAIT_API):
        record = by_handle.get(wait.event)
        require_scope(
            record is not None and record.seq < wait.seq,
            HIPScopeReason.CAUSALITY,
        )
        if record is None:
            raise HIPScopeError(HIPScopeReason.CAUSALITY)
        require_scope(
            not any(
                r.name == "hipEventDestroy"
                and r.event_handle == wait.event
                and host[record.corr].end_ns
                < r.start_ns
                < host[wait.corr].start_ns
                for r in host.values()
            ),
            HIPScopeReason.CAUSALITY,
        )
        require_scope(
            record.agent == wait.agent
            and record.queue == wait.source_queue
            and record.source_queue == record.queue
            and record.queue != wait.queue,
            HIPScopeReason.CAUSALITY,
        )
        links.append(_link(record, wait, enqueues, copies))
    return links
