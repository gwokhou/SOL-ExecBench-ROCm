# SPDX-License-Identifier: Apache-2.0
"""Cross-check SDK callback submissions against completion and export channels."""

from collections import Counter
from itertools import pairwise

from sol_execbench.core.bench.rocm_profiler.hip_causality_json import (
    SDKCapture,
    normalize_host_call,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_artifacts import (
    HIPScopeError,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    HIPScopeReason,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_models import (
    SDKAPI,
    HIPSDKLedger,
    SDKBoundary,
    SDKComplete,
    SDKDispatch,
    SDKEnqueue,
    SDKEvent,
    SDKEventComplete,
    SDKEventEnqueue,
)


def require_scope(condition: bool, reason: HIPScopeReason) -> None:
    """Never return a qualified artifact after an unmet condition."""
    if not condition:
        raise HIPScopeError(reason)


def validate_boundaries(
    ledger: HIPSDKLedger, capture: SDKCapture
) -> tuple[SDKBoundary, SDKBoundary, SDKBoundary]:
    """Bind submission tags to a single thread and successful sync fences."""
    rows = ledger.records
    require_scope(
        ledger.row_count == len(rows)
        and [r.seq for r in rows] == list(range(1, len(rows) + 1)),
        HIPScopeReason.JOURNAL,
    )
    boundaries = [r for r in rows if isinstance(r, SDKBoundary)]
    require_scope(
        [r.kind for r in boundaries] == ["begin", "end", "seal"],
        HIPScopeReason.FENCE,
    )
    begin, end, seal = boundaries
    require_scope(
        capture.metadata.init_time
        < begin.timestamp
        < end.timestamp
        < seal.timestamp
        < capture.metadata.fini_time
        and begin.tid == end.tid == seal.tid,
        HIPScopeReason.FENCE,
    )
    for row in rows:
        if isinstance(row, (SDKAPI, SDKEnqueue, SDKEventEnqueue)):
            require_scope(
                row.tid == begin.tid
                and row.scope == int(begin.seq < row.seq < end.seq),
                HIPScopeReason.JOURNAL,
            )
    _validate_submission_order(ledger, capture, begin, end)
    _validate_sync_fences(ledger, capture, boundaries)
    return begin, end, seal


def _validate_submission_order(
    ledger: HIPSDKLedger,
    capture: SDKCapture,
    begin: SDKBoundary,
    end: SDKBoundary,
) -> None:
    rows = ledger.records
    calls = [
        normalize_host_call(capture, r) for r in capture.buffer_records.hip_api
    ]
    by_id = {call.correlation_id: call for call in calls}
    apis = [r for r in rows if isinstance(r, SDKAPI)]
    require_scope(
        all(
            by_id[left.corr].end_ns < by_id[right.corr].start_ns
            for left, right in pairwise(apis)
            if left.corr in by_id and right.corr in by_id
        ),
        HIPScopeReason.JOURNAL,
    )
    for row in rows:
        if isinstance(row, (SDKEnqueue, SDKEventEnqueue)):
            following = next((r for r in apis if r.seq > row.seq), None)
            require_scope(
                following is not None and following.corr == row.corr,
                HIPScopeReason.JOURNAL,
            )
        if isinstance(row, SDKAPI) and row.corr in by_id:
            call = by_id[row.corr]
            inside = (
                begin.timestamp < call.start_ns <= call.end_ns < end.timestamp
            )
            require_scope(inside == bool(row.scope), HIPScopeReason.FENCE)


def _validate_sync_fences(
    ledger: HIPSDKLedger, capture: SDKCapture, boundaries: list[SDKBoundary]
) -> None:
    rows = ledger.records
    by_id = {
        r.correlation_id.internal: normalize_host_call(capture, r)
        for r in capture.buffer_records.hip_api
    }
    for boundary in boundaries:
        prior = [
            r
            for r in rows
            if isinstance(r, SDKAPI)
            and r.seq < boundary.seq
            and r.name != "hipGetDevice"
        ]
        require_scope(bool(prior), HIPScopeReason.FENCE)
        last = prior[-1]
        call = by_id.get(last.corr)
        require_scope(
            last.name == "hipDeviceSynchronize"
            and last.result == 0
            and call is not None
            and call.end_ns < boundary.timestamp,
            HIPScopeReason.FENCE,
        )
        require_scope(
            not any(
                isinstance(r, (SDKEnqueue, SDKEventEnqueue))
                and last.seq < r.seq < boundary.seq
                for r in rows
            ),
            HIPScopeReason.FENCE,
        )


def validate_apis(
    ledger: HIPSDKLedger, capture: SDKCapture, seal: SDKBoundary
) -> None:
    """Compare all host API exits before explicit seal, without name filtering."""
    calls = [
        normalize_host_call(capture, r) for r in capture.buffer_records.hip_api
    ]
    submitted = [r for r in ledger.records if isinstance(r, SDKAPI)]
    exported = [r for r in calls if r.end_ns < seal.timestamp]
    require_scope(
        len({r.corr for r in submitted}) == len(submitted)
        and len({r.correlation_id for r in calls}) == len(calls)
        and Counter((r.corr, r.tid, r.name, r.result) for r in submitted)
        == Counter(
            (r.correlation_id, r.thread_id, r.name, r.result) for r in exported
        ),
        HIPScopeReason.JOURNAL,
    )
    # Querying capture state is not a graph launch. Actual unsupported domains
    # fail closed even if they happen outside the selected workload range.
    require_scope(
        not any(
            token in call.name
            for call in calls
            for token in ("Graph", "BeginCapture", "EndCapture", "Ipc", "_spt")
        ),
        HIPScopeReason.INVALID,
    )


def dispatch_key(row: SDKDispatch) -> tuple[int, ...]:
    """Use dispatch ID, not host correlation ID, as the unique dispatch key."""
    return (row.dispatch, row.corr, row.tid, row.agent, row.queue, row.kernel)


def validate_dispatches(
    ledger: HIPSDKLedger, capture: SDKCapture, agent: int
) -> list[SDKEnqueue]:
    """Require whole-capture bijection; implicit dispatches are never dropped."""
    enqueues = [r for r in ledger.records if isinstance(r, SDKEnqueue)]
    complete = [r for r in ledger.records if isinstance(r, SDKComplete)]
    exported = capture.buffer_records.kernel_dispatch
    require_scope(
        bool(enqueues)
        and len({r.dispatch for r in enqueues}) == len(enqueues)
        and len({r.dispatch for r in complete}) == len(complete)
        and Counter(map(dispatch_key, enqueues))
        == Counter(map(dispatch_key, complete))
        == Counter(
            (
                r.dispatch_info.dispatch_id,
                r.correlation_id.internal,
                r.thread_id,
                r.dispatch_info.agent_id.handle,
                r.dispatch_info.queue_id.handle,
                r.dispatch_info.kernel_id,
            )
            for r in exported
        )
        and all(r.agent == agent for r in enqueues),
        HIPScopeReason.DISPATCH,
    )
    timestamps = {r.dispatch: (r.start, r.end) for r in complete}
    require_scope(
        all(
            timestamps[r.dispatch_info.dispatch_id]
            == (r.start_timestamp, r.end_timestamp)
            for r in exported
        ),
        HIPScopeReason.DISPATCH,
    )
    return enqueues


def event_key(row: SDKEvent) -> tuple[int, ...]:
    """Event correlation identifies one enqueued record/wait operation."""
    return (
        row.corr,
        row.tid,
        row.operation,
        row.event,
        row.agent,
        row.queue,
        row.source_queue,
    )


def validate_events(ledger: HIPSDKLedger, capture: SDKCapture) -> None:
    """No-op host waits remain unresolved; do not invent missing barriers."""
    enqueues = [r for r in ledger.records if isinstance(r, SDKEventEnqueue)]
    complete = [r for r in ledger.records if isinstance(r, SDKEventComplete)]
    exported = capture.buffer_records.hip_event
    require_scope(
        len({r.corr for r in enqueues}) == len(enqueues)
        and len({r.corr for r in complete}) == len(complete)
        and Counter(map(event_key, enqueues))
        == Counter(map(event_key, complete))
        == Counter(
            (
                r.correlation_id.internal,
                r.thread_id,
                r.operation,
                r.hip_event_handle,
                r.agent_id.handle,
                r.queue_id.handle,
                r.source_queue_id.handle,
            )
            for r in exported
        ),
        HIPScopeReason.BARRIER,
    )
    timestamps = {r.corr: (r.start, r.end) for r in complete}
    require_scope(
        all(
            timestamps[r.correlation_id.internal]
            == (r.start_timestamp, r.end_timestamp)
            for r in exported
        ),
        HIPScopeReason.BARRIER,
    )
