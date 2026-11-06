# SPDX-License-Identifier: Apache-2.0
"""Typed observations and scoped HIP event dispatch links."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.rocm_profiler.hip_events import HIPEventRecord
from sol_execbench.core.data.base_model import BaseModelWithDocstrings

MAX_CAUSAL_RECORDS = 10_000
RECORD_APIS = frozenset({"hipEventRecord", "hipEventRecordWithFlags"})
WAIT_API = "hipStreamWaitEvent"
DESTROY_API = "hipEventDestroy"
EVENT_SYNC_API = "hipEventSynchronize"
LAUNCH_APIS = frozenset(
    {"hipLaunchKernel", "hipModuleLaunchKernel", "hipExtModuleLaunchKernel"}
)
MEMCPY_APIS = frozenset(
    {
        "hipMemcpyAsync",
        "hipMemcpyWithStream",
        "hipMemcpy2DAsync",
        "hipMemcpy3DAsync",
    }
)
MEMCPY_DIRECTIONS = frozenset(
    {
        "MEMORY_COPY_HOST_TO_DEVICE",
        "MEMORY_COPY_DEVICE_TO_HOST",
        "MEMORY_COPY_DEVICE_TO_DEVICE",
    }
)


class HIPLinkReason(StrEnum):
    """Reasons why observations cannot establish a dispatch link."""

    INVALID = "hip_causality_artifact_invalid"
    LIMIT = "hip_causality_parse_limit"
    MISSING_HOST = "hip_causality_host_call_missing_or_ambiguous"
    HOST_MISMATCH = "hip_causality_host_barrier_mismatch"
    UNSUPPORTED = "hip_causality_unsupported_capture"
    AMBIGUOUS_ORDER = "hip_causality_host_order_ambiguous"
    MISSING_RECORD = "hip_causality_record_unresolved"
    IN_FLIGHT_RECORD = "hip_causality_record_reused_before_completion"
    MISSING_DISPATCH = "hip_causality_dispatch_endpoint_unresolved"
    DEVICE = "hip_causality_device_or_lane_mismatch"
    DUPLICATE = "hip_causality_duplicate_identity"
    CYCLE = "hip_causality_cycle"


class HIPCausalityLimitation(StrEnum):
    """Persistent evidence boundaries, including when every wait is linked."""

    COMPLETENESS = "trace_completeness_not_independently_proven"
    SCOPE = "dispatch_scope_not_bound_to_workload"
    AUTHORITY = "not_timing_or_optimization_authority"


class HIPHostCall(BaseModelWithDocstrings):
    """One successful or failed host call, never ordered by correlation ID."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    correlation_id: int = Field(gt=0)
    thread_id: int = Field(ge=0)
    name: str
    stream_id: int = Field(ge=0)
    event_handle: int | None = Field(default=None, gt=0)
    flags: int = 0
    result: int
    start_ns: int = Field(ge=0)
    end_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_interval(self) -> HIPHostCall:
        """Reject malformed host call intervals."""
        if self.end_ns <= self.start_ns:
            raise ValueError("host call interval must be positive")
        return self


class HIPCorrelatedBarrier(BaseModelWithDocstrings):
    """GPU observation retaining its host API correlation identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    event: HIPEventRecord
    correlation_id: int = Field(gt=0)
    thread_id: int = Field(ge=0)


class HIPDispatchObservation(BaseModelWithDocstrings):
    """Capture-local dispatch identity; durations are not prediction inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    dispatch_id: int = Field(gt=0)
    correlation_id: int = Field(gt=0)
    thread_id: int = Field(ge=0)
    agent_id: int = Field(ge=0)
    queue_id: int = Field(ge=0)
    stream_id: int = Field(ge=0)


class HIPEventDispatchLink(BaseModelWithDocstrings):
    """Host-submission dependency conditional on the observed capture."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    predecessor_dispatch_id: int = Field(gt=0)
    successor_dispatch_id: int = Field(gt=0)
    predecessor_correlation_id: int = Field(gt=0)
    successor_correlation_id: int = Field(gt=0)
    record_correlation_id: int = Field(gt=0)
    wait_correlation_id: int = Field(gt=0)
    event_handle: int = Field(gt=0)
    agent_id: int = Field(ge=0)
    source_queue_id: int = Field(ge=0)
    destination_queue_id: int = Field(ge=0)
    source_stream_id: int = Field(gt=0)
    destination_stream_id: int = Field(gt=0)
    basis: Literal["hip_host_submission_order"] = "hip_host_submission_order"


class HIPUnresolvedWait(BaseModelWithDocstrings):
    """An observed host wait that could not be linked, including no-op waits."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    correlation_id: int = Field(gt=0)
    reason: HIPLinkReason = Field(strict=False)


class HIPEventCausality(BaseModelWithDocstrings):
    """Partial diagnostic topology; never a complete workload schedule."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    capture: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    process_id: int = Field(ge=0)
    status: Literal["partial"] = "partial"
    links: list[HIPEventDispatchLink] = Field(default_factory=list)
    unresolved_waits: list[HIPUnresolvedWait] = Field(default_factory=list)
    reason_codes: list[Annotated[HIPLinkReason, Field(strict=False)]] = Field(
        default_factory=list
    )
    limitations: tuple[
        Annotated[HIPCausalityLimitation, Field(strict=False)], ...
    ] = Field(default=tuple(HIPCausalityLimitation), strict=False)
