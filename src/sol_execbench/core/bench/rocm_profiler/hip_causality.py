# SPDX-License-Identifier: Apache-2.0
"""Join host event calls to GPU barriers and dispatch submission identities."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from sol_execbench.core.bench.rocm_profiler.hip_causality_models import (
    DESTROY_API,
    EVENT_SYNC_API,
    LAUNCH_APIS,
    RECORD_APIS,
    WAIT_API,
    HIPCorrelatedBarrier,
    HIPDispatchObservation,
    HIPEventCausality,
    HIPEventDispatchLink,
    HIPHostCall,
    HIPLinkReason,
    HIPUnresolvedWait,
)
from sol_execbench.core.bench.rocm_profiler.hip_events import HIPEventOperation


@dataclass(frozen=True, slots=True, kw_only=True)
class HIPCausalTrace:
    """One bounded capture, with no joins across files or processes."""

    capture: str
    artifact_sha256: str
    process_id: int
    calls: tuple[HIPHostCall, ...]
    barriers: tuple[HIPCorrelatedBarrier, ...]
    dispatches: tuple[HIPDispatchObservation, ...]


class HIPLinkError(ValueError):
    """A stable rejection reason, not a fallback to timestamp inference."""

    def __init__(self, reason: HIPLinkReason) -> None:
        """Retain the machine-readable reason without raw trace contents."""
        super().__init__(reason.value)
        self.reason = reason


def correlate_hip_events(trace: HIPCausalTrace) -> HIPEventCausality:
    """Return conditional links and explicit unresolved waits for one capture."""
    result = HIPEventCausality(
        capture=trace.capture,
        artifact_sha256=trace.artifact_sha256,
        process_id=trace.process_id,
    )
    waits = [call for call in trace.calls if call.name == WAIT_API]
    try:
        _unique_identities(trace)
    except HIPLinkError as error:
        return result.model_copy(update={"reason_codes": [error.reason]})
    calls = {call.correlation_id: call for call in trace.calls}
    barriers = {item.correlation_id: item for item in trace.barriers}
    if set(barriers) - set(calls):
        return result.model_copy(
            update={
                "reason_codes": [HIPLinkReason.MISSING_HOST],
                "unresolved_waits": [
                    HIPUnresolvedWait(
                        correlation_id=wait.correlation_id,
                        reason=HIPLinkReason.MISSING_HOST,
                    )
                    for wait in waits
                ],
            }
        )
    for wait in sorted(waits, key=lambda item: item.start_ns):
        try:
            link = _wait_link(trace, wait, calls, barriers)
        except HIPLinkError as error:
            result.unresolved_waits.append(
                HIPUnresolvedWait(
                    correlation_id=wait.correlation_id, reason=error.reason
                )
            )
        else:
            result.links.append(link)
    if _cyclic(result.links):
        return result.model_copy(
            update={"links": [], "reason_codes": [HIPLinkReason.CYCLE]}
        )
    return result


def _unique_identities(trace: HIPCausalTrace) -> None:
    for values in (
        [call.correlation_id for call in trace.calls],
        [barrier.correlation_id for barrier in trace.barriers],
        [dispatch.dispatch_id for dispatch in trace.dispatches],
        [dispatch.correlation_id for dispatch in trace.dispatches],
    ):
        if len(values) != len(set(values)):
            raise HIPLinkError(HIPLinkReason.DUPLICATE)
    if any(
        item.event.capture != trace.capture
        or item.event.process_id != trace.process_id
        for item in trace.barriers
    ):
        raise HIPLinkError(HIPLinkReason.DEVICE)


def _matched_barrier(
    call: HIPHostCall, barriers: dict[int, HIPCorrelatedBarrier]
) -> HIPCorrelatedBarrier:
    barrier = barriers.get(call.correlation_id)
    if barrier is None:
        raise HIPLinkError(HIPLinkReason.MISSING_HOST)
    expected = (
        HIPEventOperation.WAIT
        if call.name == WAIT_API
        else HIPEventOperation.RECORD
    )
    event = barrier.event
    if (
        event.operation != expected
        or event.event_handle != call.event_handle
        or event.stream_id != call.stream_id
        or barrier.thread_id != call.thread_id
        or call.result != 0
        or call.flags != 0
        or call.stream_id == 0
    ):
        raise HIPLinkError(HIPLinkReason.HOST_MISMATCH)
    return barrier


def _record_for_wait(
    trace: HIPCausalTrace,
    wait: HIPHostCall,
    barriers: dict[int, HIPCorrelatedBarrier],
) -> HIPHostCall:
    history = sorted(
        (
            call
            for call in trace.calls
            if call.event_handle == wait.event_handle
        ),
        key=lambda item: item.start_ns,
    )
    if len({call.thread_id for call in history}) != 1 or any(
        left.end_ns >= right.start_ns for left, right in pairwise(history)
    ):
        raise HIPLinkError(HIPLinkReason.AMBIGUOUS_ORDER)
    before = [call for call in history if call.end_ns < wait.start_ns]
    record = None
    for call in before:
        if call.name == DESTROY_API:
            record = None
        elif call.name in RECORD_APIS:
            current = _matched_barrier(call, barriers)
            if record is not None:
                previous = _matched_barrier(record, barriers)
                if previous.event.end_ns >= call.start_ns and not any(
                    sync.name == EVENT_SYNC_API
                    and sync.result == 0
                    and record.end_ns
                    < sync.start_ns
                    < sync.end_ns
                    < call.start_ns
                    for sync in history
                ):
                    raise HIPLinkError(HIPLinkReason.IN_FLIGHT_RECORD)
                if previous.event.agent_id != current.event.agent_id:
                    raise HIPLinkError(HIPLinkReason.DEVICE)
            record = call
    if record is None:
        raise HIPLinkError(HIPLinkReason.MISSING_RECORD)
    return record


def _endpoint(
    trace: HIPCausalTrace,
    call: HIPHostCall,
    barrier: HIPCorrelatedBarrier,
    calls: dict[int, HIPHostCall],
    *,
    before: bool,
) -> tuple[HIPDispatchObservation, HIPHostCall]:
    matches = []
    event = barrier.event
    for dispatch in trace.dispatches:
        if (dispatch.agent_id, dispatch.queue_id, dispatch.stream_id) != (
            event.agent_id,
            event.queue_id,
            event.stream_id,
        ):
            continue
        host = calls.get(dispatch.correlation_id)
        if host is None:
            raise HIPLinkError(HIPLinkReason.MISSING_HOST)
        if (
            host.name not in LAUNCH_APIS
            or host.result != 0
            or host.stream_id != dispatch.stream_id
            or host.thread_id != dispatch.thread_id
            or host.thread_id != call.thread_id
        ):
            raise HIPLinkError(HIPLinkReason.HOST_MISMATCH)
        if host.start_ns <= call.end_ns and host.end_ns >= call.start_ns:
            raise HIPLinkError(HIPLinkReason.AMBIGUOUS_ORDER)
        if (before and host.end_ns < call.start_ns) or (
            not before and host.start_ns > call.end_ns
        ):
            matches.append((dispatch, host))
    if not matches:
        raise HIPLinkError(HIPLinkReason.MISSING_DISPATCH)
    matches.sort(key=lambda item: item[1].start_ns)
    if any(
        left[1].end_ns >= right[1].start_ns for left, right in pairwise(matches)
    ):
        raise HIPLinkError(HIPLinkReason.AMBIGUOUS_ORDER)
    return matches[-1 if before else 0]


def _wait_link(
    trace: HIPCausalTrace,
    wait: HIPHostCall,
    calls: dict[int, HIPHostCall],
    barriers: dict[int, HIPCorrelatedBarrier],
) -> HIPEventDispatchLink:
    waiting = _matched_barrier(wait, barriers)
    record = _record_for_wait(trace, wait, barriers)
    recorded = _matched_barrier(record, barriers)
    if (
        recorded.event.agent_id != waiting.event.agent_id
        or recorded.event.queue_id != waiting.event.source_queue_id
    ):
        raise HIPLinkError(HIPLinkReason.DEVICE)
    predecessor, source = _endpoint(trace, record, recorded, calls, before=True)
    successor, destination = _endpoint(
        trace, wait, waiting, calls, before=False
    )
    return HIPEventDispatchLink(
        predecessor_dispatch_id=predecessor.dispatch_id,
        successor_dispatch_id=successor.dispatch_id,
        predecessor_correlation_id=source.correlation_id,
        successor_correlation_id=destination.correlation_id,
        record_correlation_id=record.correlation_id,
        wait_correlation_id=wait.correlation_id,
        event_handle=recorded.event.event_handle,
        agent_id=recorded.event.agent_id,
        source_queue_id=recorded.event.queue_id,
        destination_queue_id=waiting.event.queue_id,
        source_stream_id=record.stream_id,
        destination_stream_id=wait.stream_id,
    )


def _cyclic(links: list[HIPEventDispatchLink]) -> bool:
    children: dict[int, set[int]] = defaultdict(set)
    indegree: dict[int, int] = defaultdict(int)
    for link in links:
        source, target = (
            link.predecessor_dispatch_id,
            link.successor_dispatch_id,
        )
        indegree.setdefault(source, 0)
        if target not in children[source]:
            children[source].add(target)
            indegree[target] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    seen = 0
    while ready:
        seen += 1
        for child in children[ready.pop()]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    return seen != len(indegree)
