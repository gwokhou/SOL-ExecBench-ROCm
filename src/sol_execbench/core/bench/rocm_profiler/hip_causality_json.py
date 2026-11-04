# SPDX-License-Identifier: Apache-2.0
"""Validate the ROCprofiler JSON fields required for host-order correlation."""

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.bench.rocm_profiler.hip_causality import HIPCausalTrace
from sol_execbench.core.bench.rocm_profiler.hip_causality_models import (
    DESTROY_API,
    EVENT_SYNC_API,
    MAX_CAUSAL_RECORDS,
    RECORD_APIS,
    WAIT_API,
    HIPCorrelatedBarrier,
    HIPDispatchObservation,
    HIPHostCall,
)
from sol_execbench.core.bench.rocm_profiler.hip_events import (
    HIPEventOperation,
    HIPEventRecord,
)

MAX_CAUSAL_CAPTURES = 64
MAX_SDK_API_RECORDS = 100_000


class _SDKModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class _Handle(_SDKModel):
    handle: int = Field(ge=0)


class _Correlation(_SDKModel):
    internal: int = Field(gt=0)


class _Argument(_SDKModel):
    name: str
    value: str


class _Return(_SDKModel):
    uint64_t_retval: int


class _Activity(_SDKModel):
    kind: int = Field(ge=0)
    operation: int = Field(ge=0)
    thread_id: int = Field(ge=0)
    correlation_id: _Correlation
    start_timestamp: int = Field(ge=0)
    end_timestamp: int = Field(ge=0)
    stream_id: _Handle


class _Call(_Activity):
    args: list[_Argument]
    retval: _Return


class _Barrier(_Activity):
    agent_id: _Handle
    queue_id: _Handle
    hip_event_handle: int = Field(gt=0)
    source_queue_id: _Handle


class _DispatchInfo(_SDKModel):
    kernel_id: int | None = Field(default=None, gt=0)
    dispatch_id: int = Field(gt=0)
    agent_id: _Handle
    queue_id: _Handle


class _Dispatch(_Activity):
    dispatch_info: _DispatchInfo
    graph_exec_id: int = 0
    graph_node_id: int = 0


class _Records(_SDKModel):
    hip_api: list[_Call] = Field(
        default_factory=list, max_length=MAX_SDK_API_RECORDS
    )
    hip_event: list[_Barrier] = Field(
        default_factory=list, max_length=MAX_CAUSAL_RECORDS
    )
    kernel_dispatch: list[_Dispatch] = Field(
        default_factory=list, max_length=MAX_CAUSAL_RECORDS
    )


class _Metadata(_SDKModel):
    command: list[str] = Field(default_factory=list)
    pid: int = Field(ge=0)
    init_time: int = Field(ge=0)
    fini_time: int = Field(gt=0)


class _Kind(_SDKModel):
    kind: str
    operations: list[str]


class _Strings(_SDKModel):
    buffer_records: list[_Kind]


class SDKCapture(_SDKModel):
    """Validated source capture; unrelated SDK fields are not interpreted."""

    metadata: _Metadata
    strings: _Strings
    buffer_records: _Records

    def operation(self, item: _Activity, kind: str) -> str:
        """Resolve operation names without hard-coded SDK enum ordinals."""
        table = self.strings.buffer_records[item.kind]
        if table.kind != kind:
            raise ValueError("incorrect SDK tracing domain")
        return table.operations[item.operation]


class SDKCausalDocument(_SDKModel):
    """A bounded collection of independent process captures."""

    captures: list[SDKCapture] = Field(
        alias="rocprofiler-sdk-tool", max_length=MAX_CAUSAL_CAPTURES
    )


def normalize_host_call(capture: SDKCapture, call: _Call) -> HIPHostCall:
    """Normalize one API row; the caller must validate capture completeness."""
    name = capture.operation(call, "HIP_RUNTIME_API_EXT")
    arguments = {item.name: item.value for item in call.args}
    if len(arguments) != len(call.args):
        raise ValueError("duplicate API argument")
    # Create APIs expose the address of an output pointer, not the event handle.
    event = arguments.get("event") if "Create" not in name else None
    if (
        name in RECORD_APIS | {WAIT_API, DESTROY_API, EVENT_SYNC_API}
        and event is None
    ):
        raise ValueError("event argument is missing")
    if (
        name in {WAIT_API, "hipEventRecordWithFlags"}
        and "flags" not in arguments
    ):
        raise ValueError("flags argument is missing")
    return HIPHostCall(
        correlation_id=call.correlation_id.internal,
        thread_id=call.thread_id,
        name=name,
        stream_id=call.stream_id.handle,
        event_handle=int(event, 0) if event is not None else None,
        flags=int(arguments.get("flags", "0"), 0),
        result=call.retval.uint64_t_retval,
        start_ns=call.start_timestamp,
        end_ns=call.end_timestamp,
    )


def causal_trace(
    capture: SDKCapture, label: str, digest: str
) -> HIPCausalTrace:
    """Normalize one capture after validating process lifetime and domains."""
    records = capture.buffer_records
    if (
        sum(
            map(
                len,
                (records.hip_api, records.hip_event, records.kernel_dispatch),
            )
        )
        > MAX_CAUSAL_RECORDS
    ):
        raise ValueError("combined capture row limit")
    if capture.metadata.init_time >= capture.metadata.fini_time:
        raise ValueError("capture did not finish")
    for item in (
        *records.hip_api,
        *records.hip_event,
        *records.kernel_dispatch,
    ):
        if (
            not capture.metadata.init_time
            <= item.start_timestamp
            <= item.end_timestamp
            <= capture.metadata.fini_time
        ):
            raise ValueError("activity outside process lifetime")
    calls = tuple(
        normalize_host_call(capture, item) for item in records.hip_api
    )
    barriers = tuple(
        HIPCorrelatedBarrier(
            correlation_id=item.correlation_id.internal,
            thread_id=item.thread_id,
            event=HIPEventRecord(
                capture=label,
                process_id=capture.metadata.pid,
                agent_id=item.agent_id.handle,
                queue_id=item.queue_id.handle,
                stream_id=item.stream_id.handle,
                event_handle=item.hip_event_handle,
                source_queue_id=item.source_queue_id.handle,
                operation=HIPEventOperation(
                    capture.operation(item, "HIP_EVENT")
                ),
                start_ns=item.start_timestamp,
                end_ns=item.end_timestamp,
            ),
        )
        for item in records.hip_event
    )
    dispatches = tuple(
        _dispatch(capture, item) for item in records.kernel_dispatch
    )
    return HIPCausalTrace(
        label, digest, capture.metadata.pid, calls, barriers, dispatches
    )


def _dispatch(capture: SDKCapture, item: _Dispatch) -> HIPDispatchObservation:
    if capture.operation(item, "KERNEL_DISPATCH") != "KERNEL_DISPATCH_COMPLETE":
        raise ValueError("unsupported dispatch record")
    if item.graph_exec_id != 0 or item.graph_node_id != 0:
        raise ValueError("HIP graph dispatch is not qualified")
    return HIPDispatchObservation(
        dispatch_id=item.dispatch_info.dispatch_id,
        correlation_id=item.correlation_id.internal,
        thread_id=item.thread_id,
        agent_id=item.dispatch_info.agent_id.handle,
        queue_id=item.dispatch_info.queue_id.handle,
        stream_id=item.stream_id.handle,
    )
