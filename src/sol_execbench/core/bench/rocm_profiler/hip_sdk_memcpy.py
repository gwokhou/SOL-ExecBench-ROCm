# SPDX-License-Identifier: Apache-2.0
"""Engine-copy domain: three-channel agreement plus co-observed lane binding."""

from collections import Counter, defaultdict

from sol_execbench.core.bench.rocm_profiler.hip_causality_json import (
    SDKCapture,
    SDKCopyRecord,
    normalize_host_call,
)
from sol_execbench.core.bench.rocm_profiler.hip_causality_models import (
    MEMCPY_APIS,
    MEMCPY_DIRECTIONS,
    HIPHostCall,
)

ONE_DIMENSIONAL_MEMCPY_APIS = frozenset(
    {"hipMemcpyAsync", "hipMemcpyWithStream"}
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_artifacts import (
    HIPScopeError,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    HIPScopeReason,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_models import (
    SDKAPI,
    HIPSDKCopyDomain,
    HIPSDKCopyDomainReason,
    HIPSDKEngineCopy,
    HIPSDKLedger,
    SDKBoundary,
    SDKMemcpy,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_validation import (
    require_scope,
)

DEFAULT_STREAM_ORDINAL = 0


def memcpy_copy_domain(
    ledger: HIPSDKLedger,
    capture: SDKCapture,
    begin: SDKBoundary,
    end: SDKBoundary,
) -> HIPSDKCopyDomain:
    """Fail closed on channel disagreement; vocabulary gaps stay reason codes."""
    copies = [r for r in ledger.records if isinstance(r, SDKMemcpy)]
    exported = capture.buffer_records.memory_copy
    apis = {r.corr: r for r in ledger.records if isinstance(r, SDKAPI)}
    hosts = {
        call.correlation_id: call
        for call in (
            normalize_host_call(capture, r)
            for r in capture.buffer_records.hip_api
        )
    }
    args = _argument_table(capture)
    _require_copy_channels(copies, exported, apis, hosts, args, capture)
    return _copy_domain(
        copies, exported, apis, hosts, args, begin, end, capture
    )


def _argument_table(capture: SDKCapture) -> dict[int, dict[str, str]]:
    """Argument strings keyed by correlation; duplicate names are channel lies."""
    table: dict[int, dict[str, str]] = {}
    for row in capture.buffer_records.hip_api:
        arguments = {item.name: item.value for item in row.args}
        require_scope(
            len(arguments) == len(row.args)
            and row.correlation_id.internal not in table,
            HIPScopeReason.COPY,
        )
        table[row.correlation_id.internal] = arguments
    return table


def _require_copy_channels(
    copies: list[SDKMemcpy],
    exported: list[SDKCopyRecord],
    apis: dict[int, SDKAPI],
    hosts: dict[int, HIPHostCall],
    args: dict[int, dict[str, str]],
    capture: SDKCapture,
) -> None:
    """Whole-capture bijection between ledger and export copy channels.

    One host call may complete as several engine transfers (chunked large
    copies share one correlation), so channels agree as multisets and the
    chunk bytes of a 1D call must sum to its host sizeBytes.
    """
    _require_copy_bijection(copies, exported)
    for row in exported:
        require_scope(
            capture.metadata.init_time
            <= row.start_timestamp
            <= row.end_timestamp
            <= capture.metadata.fini_time,
            HIPScopeReason.COPY,
        )
        api = apis.get(row.correlation_id.internal)
        host = hosts.get(row.correlation_id.internal)
        require_scope(
            api is not None and host is not None,
            HIPScopeReason.COPY,
        )
        if api is None or host is None:
            raise HIPScopeError(HIPScopeReason.COPY)
        require_scope(
            api.name == host.name
            and api.result == host.result == 0
            and api.tid == host.thread_id,
            HIPScopeReason.COPY,
        )
        arguments = args.get(row.correlation_id.internal, {})
        if api.name in ONE_DIMENSIONAL_MEMCPY_APIS and "sizeBytes" in arguments:
            chunks = sum(
                r.bytes
                for r in exported
                if r.correlation_id.internal == row.correlation_id.internal
            )
            require_scope(
                chunks == int(arguments["sizeBytes"], 0),
                HIPScopeReason.COPY,
            )
        require_scope(
            api.name not in MEMCPY_APIS
            or "stream" not in arguments
            or api.stream == int(arguments["stream"], 0),
            HIPScopeReason.COPY,
        )


def _copy_domain(
    copies: list[SDKMemcpy],
    exported: list[SDKCopyRecord],
    apis: dict[int, SDKAPI],
    hosts: dict[int, HIPHostCall],
    args: dict[int, dict[str, str]],
    begin: SDKBoundary,
    end: SDKBoundary,
    capture: SDKCapture,
) -> HIPSDKCopyDomain:
    directions = {
        row.correlation_id.internal: capture.operation(row, "MEMORY_COPY")
        for row in exported
    }
    queues_by_ordinal: dict[int, set[int]] = defaultdict(set)
    for row in capture.buffer_records.kernel_dispatch:
        queues_by_ordinal[row.stream_id.handle].add(
            row.dispatch_info.queue_id.handle
        )
    reasons: set[HIPSDKCopyDomainReason] = set()
    bound: list[HIPSDKEngineCopy] = []
    outside: list[int] = []
    for row in copies:
        api = apis[row.corr]
        ordinal = hosts[row.corr].stream_id
        handle = _stream_handle(row.corr, api, args)
        queue = None
        in_scope = begin.seq < api.seq < end.seq
        if in_scope:
            if api.name not in MEMCPY_APIS:
                reasons.add(HIPSDKCopyDomainReason.UNSUPPORTED_API)
            elif directions[row.corr] not in MEMCPY_DIRECTIONS:
                reasons.add(HIPSDKCopyDomainReason.UNSUPPORTED_DIRECTION)
            elif handle == DEFAULT_STREAM_ORDINAL:
                reasons.add(HIPSDKCopyDomainReason.DEFAULT_STREAM)
            else:
                queue = _co_observed_queue(ordinal, queues_by_ordinal)
                if queue is None:
                    reasons.update(
                        _unresolved_reasons(ordinal, queues_by_ordinal)
                    )
        elif row.corr not in outside:
            outside.append(row.corr)
        bound.append(
            HIPSDKEngineCopy(
                correlation_id=row.corr,
                direction=directions[row.corr],
                bytes=row.bytes,
                start_ns=row.start,
                end_ns=row.end,
                stream_handle=handle,
                stream_ordinal=ordinal,
                queue_id=queue,
                submission_seq=api.seq,
                in_scope=in_scope,
            )
        )
    return HIPSDKCopyDomain(
        copies=tuple(bound),
        outside_correlation_ids=tuple(sorted(outside)),
        reason_codes=tuple(sorted(reasons, key=str)),
        qualified=not reasons,
    )


def _stream_handle(
    corr: int, api: SDKAPI, args: dict[int, dict[str, str]]
) -> int | None:
    """Raw stream argument recorded by the collector for the core API set.

    None when no channel recorded a stream; never interpreted as handle zero.
    """
    if api.stream is not None:
        return api.stream
    argument = args.get(corr, {}).get("stream")
    return int(argument, 0) if argument is not None else None


def _co_observed_queue(
    ordinal: int, queues_by_ordinal: dict[int, set[int]]
) -> int | None:
    """A stream ordinal binds one queue when kernels share the stream."""
    queues = queues_by_ordinal.get(ordinal, frozenset())
    if len(queues) == 1:
        return next(iter(queues))
    return None


def _unresolved_reasons(
    ordinal: int, queues_by_ordinal: dict[int, set[int]]
) -> set[HIPSDKCopyDomainReason]:
    queues = queues_by_ordinal.get(ordinal, frozenset())
    if queues:
        return {HIPSDKCopyDomainReason.STREAM_CONFLICT}
    return {HIPSDKCopyDomainReason.QUEUE_UNRESOLVED}


__all__ = ["DEFAULT_STREAM_ORDINAL", "memcpy_copy_domain"]


def _require_copy_bijection(
    copies: list[SDKMemcpy], exported: list[SDKCopyRecord]
) -> None:
    """Match ledger and exported copy channels as complete multisets."""
    require_scope(
        Counter(
            (
                r.corr,
                r.tid,
                r.operation,
                r.src_agent,
                r.dst_agent,
                r.bytes,
                r.start,
                r.end,
            )
            for r in copies
        )
        == Counter(
            (
                r.correlation_id.internal,
                r.thread_id,
                r.operation,
                r.src_agent_id.handle,
                r.dst_agent_id.handle,
                r.bytes,
                r.start_timestamp,
                r.end_timestamp,
            )
            for r in exported
        ),
        HIPScopeReason.COPY,
    )
