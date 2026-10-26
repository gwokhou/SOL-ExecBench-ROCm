# SPDX-License-Identifier: Apache-2.0
"""Capture coverage shared by audited probes and production qualification."""

from sol_execbench.core.bench.rocm_profiler.hip_causality_json import SDKCapture
from sol_execbench.core.bench.rocm_profiler.hip_causality_models import (
    LAUNCH_APIS,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_artifacts import (
    HIPScopeCaptureView,
    HIPScopeError,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    HIPScopeReason,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_models import (
    HIPSDKBoundDispatch,
    HIPSDKLedger,
    SDKBoundary,
    SDKEnqueue,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_validation import (
    require_scope,
)


def validate_lifetime(
    capture: SDKCapture,
    ledger: HIPSDKLedger,
    begin: SDKBoundary,
    end: SDKBoundary,
    seal: SDKBoundary,
) -> None:
    """Require bounded eager activity and an audited teardown suffix."""
    records = capture.buffer_records
    require_scope(
        all(
            capture.metadata.init_time
            <= r.start_timestamp
            <= r.end_timestamp
            <= capture.metadata.fini_time
            for r in (
                *records.hip_api,
                *records.hip_event,
                *records.kernel_dispatch,
            )
        ),
        HIPScopeReason.FENCE,
    )
    scoped = {
        r.dispatch
        for r in ledger.records
        if isinstance(r, SDKEnqueue) and r.scope == 1
    }
    require_scope(bool(scoped), HIPScopeReason.DISPATCH)
    for row in records.kernel_dispatch:
        require_scope(
            row.graph_exec_id == row.graph_node_id == 0
            and row.end_timestamp < seal.timestamp
            and capture.operation(row, "KERNEL_DISPATCH")
            == "KERNEL_DISPATCH_COMPLETE",
            HIPScopeReason.DISPATCH,
        )
        if row.dispatch_info.dispatch_id in scoped:
            require_scope(
                begin.timestamp
                < row.start_timestamp
                <= row.end_timestamp
                < end.timestamp,
                HIPScopeReason.FENCE,
            )
    # Host calls after seal are retained in the original trace but not covered
    # by the independent API ledger. Only audited interpreter teardown is allowed.
    allowed = frozenset(
        {
            "hipModuleUnload",
            "hipFree",
            "hipEventDestroy",
            "hipStreamDestroy",
            "hipGetDevice",
        }
    )
    require_scope(
        all(
            capture.operation(r, "HIP_RUNTIME_API_EXT") in allowed
            for r in records.hip_api
            if r.end_timestamp >= seal.timestamp
        ),
        HIPScopeReason.JOURNAL,
    )


def bind_dispatches(
    capture: SDKCapture,
    view: HIPScopeCaptureView,
    enqueues: list[SDKEnqueue],
) -> tuple[list[HIPSDKBoundDispatch], list[int]]:
    """Bind all in-scope enqueues, including library-internal helpers."""
    symbols = {s.kernel_id: s.kernel_name for s in view.kernel_symbols}
    require_scope(
        len(symbols) == len(view.kernel_symbols), HIPScopeReason.DISPATCH
    )
    raw = {
        r.dispatch_info.dispatch_id: r
        for r in capture.buffer_records.kernel_dispatch
    }
    calls = {
        r.correlation_id.internal: r for r in capture.buffer_records.hip_api
    }
    bound, implicit = [], []
    for row in enqueues:
        call = calls.get(row.corr)
        if call is None:
            raise HIPScopeError(HIPScopeReason.DISPATCH)
        require_scope(call.retval.uint64_t_retval == 0, HIPScopeReason.DISPATCH)
        if row.scope == 0:
            continue
        dispatch = raw[row.dispatch]
        bound.append(
            HIPSDKBoundDispatch(
                dispatch_id=row.dispatch,
                correlation_id=row.corr,
                kernel_name=symbols[row.kernel],
                agent_id=row.agent,
                queue_id=row.queue,
                stream_id=dispatch.stream_id.handle,
            )
        )
        if capture.operation(call, "HIP_RUNTIME_API_EXT") not in LAUNCH_APIS:
            implicit.append(row.dispatch)
    return bound, implicit
