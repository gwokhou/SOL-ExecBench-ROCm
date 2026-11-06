# SPDX-License-Identifier: Apache-2.0
"""Typed independent SDK ledger and bounded eager-workload qualification."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.rocm_profiler.hip_causality_json import (
    MAX_SDK_API_RECORDS,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    ROCR_UUID_PATTERN,
    SHA256_PATTERN,
    UUID_PATTERN,
    HIPBoundDispatch,
)
from sol_execbench.core.data.base_model import (
    BaseModelWithDocstrings,
    CurrentFrozenSchemaModel,
)
from sol_execbench.core.integrity.schema_versions import SchemaVersion

MAX_TORCH_SCOPE_BYTES = 64 * 1024 * 1024


class SDKLedgerRow(BaseModelWithDocstrings):
    """Collector sequence is host callback order, never GPU timestamp order."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    seq: int = Field(gt=0)
    tid: int = Field(gt=0)


class SDKBoundary(SDKLedgerRow):
    """Executor boundary observed by the SDK tool."""

    kind: Literal["begin", "end", "seal"]
    timestamp: int = Field(gt=0)


class SDKAPI(SDKLedgerRow):
    """Synchronous HIP API exit and its return value.

    The raw stream argument is recorded only for the frozen memcpy core set;
    zero is the legacy default stream and authorizes no sync inference.
    """

    kind: Literal["api"]
    scope: Literal[0, 1]
    corr: int = Field(gt=0)
    name: str = Field(min_length=1)
    result: int
    stream: int | None = Field(default=None, ge=0)


class SDKDispatch(SDKLedgerRow):
    """Dispatch identity shared by submission and completion channels."""

    corr: int = Field(gt=0)
    dispatch: int = Field(gt=0)
    agent: int = Field(ge=0)
    queue: int = Field(ge=0)
    kernel: int = Field(gt=0)


class SDKEnqueue(SDKDispatch):
    """Submitted kernel with membership determined at submission."""

    kind: Literal["enqueue"]
    scope: Literal[0, 1]


class SDKComplete(SDKDispatch):
    """Buffered kernel completion."""

    kind: Literal["complete"]
    start: int = Field(gt=0)
    end: int = Field(gt=0)


class SDKEvent(SDKLedgerRow):
    """Enqueued HIP event identity."""

    corr: int = Field(gt=0)
    operation: int = Field(gt=0)
    event: int = Field(gt=0)
    agent: int = Field(ge=0)
    queue: int = Field(ge=0)
    source_queue: int = Field(ge=0)


class SDKEventEnqueue(SDKEvent):
    """Synchronous HIP event enqueue."""

    kind: Literal["event_enqueue"]
    scope: Literal[0, 1]


class SDKEventComplete(SDKEvent):
    """Buffered HIP event completion."""

    kind: Literal["event_complete"]
    start: int = Field(gt=0)
    end: int = Field(gt=0)


class SDKMemcpy(SDKLedgerRow):
    """Buffered engine copy completion; the SDK record carries no queue."""

    kind: Literal["memcpy"]
    corr: int = Field(gt=0)
    operation: int = Field(gt=0)
    src_agent: int = Field(ge=0)
    dst_agent: int = Field(ge=0)
    bytes: int = Field(gt=0)
    start: int = Field(gt=0)
    end: int = Field(gt=0)


class HIPSDKLedger(CurrentFrozenSchemaModel):
    """Fail closed without a successful lossless-buffer finalization footer."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    current_schema_version = SchemaVersion.HIP_SDK_LEDGER
    schema_version: Literal[SchemaVersion.HIP_SDK_LEDGER]
    run_id: str = Field(pattern=UUID_PATTERN)
    pid: int = Field(gt=0)
    records: list[
        Annotated[
            SDKBoundary
            | SDKAPI
            | SDKEnqueue
            | SDKComplete
            | SDKEventEnqueue
            | SDKEventComplete
            | SDKMemcpy,
            Field(discriminator="kind"),
        ]
    ] = Field(min_length=2, max_length=MAX_SDK_API_RECORDS)
    row_count: int = Field(gt=0, le=MAX_SDK_API_RECORDS)
    drop_count: Literal[0]
    lossless: Literal[True]
    finalized: Literal[True]
    stop_ok: Literal[True]
    flush_ok: Literal[True]


class HIPTorchExpectation(BaseModelWithDocstrings):
    """Controller-owned identity fixed before launching the audited executor."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    run_id: str = Field(pattern=UUID_PATTERN)
    workload_id: str
    device_uuid: str = Field(pattern=ROCR_UUID_PATTERN)
    definition_sha256: str = Field(pattern=SHA256_PATTERN)
    executor_sha256: str = Field(pattern=SHA256_PATTERN)
    collector_sha256: str = Field(pattern=SHA256_PATTERN)
    application_command: tuple[str, ...] = Field(strict=False)
    returncode: Literal[0]


class HIPTorchReceipt(BaseModelWithDocstrings):
    """Audited executor completion, numerical check and source identities."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    run_id: str = Field(pattern=UUID_PATTERN)
    workload_id: str
    process_id: int = Field(gt=0)
    definition_sha256: str = Field(pattern=SHA256_PATTERN)
    executor_sha256: str = Field(pattern=SHA256_PATTERN)
    collector_sha256: str = Field(pattern=SHA256_PATTERN)
    device_uuid: str
    shape: list[int] = Field(min_length=3, max_length=3)
    correct: Literal[True]
    completed: Literal[True]


class HIPSDKEventLink(BaseModelWithDocstrings):
    """SDK submission endpoints: a kernel dispatch or a lane-bound engine copy."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    predecessor_dispatch_id: int | None = Field(default=None, gt=0)
    successor_dispatch_id: int | None = Field(default=None, gt=0)
    predecessor_copy_correlation_id: int | None = Field(default=None, gt=0)
    successor_copy_correlation_id: int | None = Field(default=None, gt=0)
    record_correlation_id: int = Field(gt=0)
    wait_correlation_id: int = Field(gt=0)
    event_handle: int = Field(gt=0)
    basis: Literal["sdk_enqueue_order"] = "sdk_enqueue_order"

    @model_validator(mode="after")
    def endpoints_are_exclusive(self) -> HIPSDKEventLink:
        """Exactly one endpoint kind per side; copies use correlation identity."""
        for dispatch, copy in (
            (
                self.predecessor_dispatch_id,
                self.predecessor_copy_correlation_id,
            ),
            (self.successor_dispatch_id, self.successor_copy_correlation_id),
        ):
            if (dispatch is None) == (copy is None):
                raise ValueError("event link endpoint must be dispatch or copy")
        return self


class HIPSDKCopyDomainReason(StrEnum):
    """Why the engine-copy dependency domain is not qualified."""

    UNSUPPORTED_API = "copy_api_unsupported"
    UNSUPPORTED_DIRECTION = "copy_direction_unsupported"
    QUEUE_UNRESOLVED = "copy_queue_unresolved"
    STREAM_CONFLICT = "copy_stream_conflict"
    DEFAULT_STREAM = "default_stream_implicit_sync_not_inferred"
    UNQUALIFIED = "copy_domain_unqualified"


class HIPSDKEngineCopy(BaseModelWithDocstrings):
    """One engine copy with its co-observed lane, or an explicit gap."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    correlation_id: int = Field(gt=0)
    direction: str = Field(min_length=1)
    bytes: int = Field(gt=0)
    start_ns: int = Field(gt=0)
    end_ns: int = Field(gt=0)
    stream_handle: int | None = Field(default=None, ge=0)
    stream_ordinal: int = Field(ge=0)
    queue_id: int | None = Field(default=None, ge=0)
    submission_seq: int = Field(gt=0)
    in_scope: bool


class HIPSDKCopyDomain(BaseModelWithDocstrings):
    """Engine-copy domain result; kernel-lowered copies stay dispatches."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    copies: tuple[HIPSDKEngineCopy, ...] = Field(default=(), strict=False)
    outside_correlation_ids: tuple[int, ...] = Field(default=(), strict=False)
    reason_codes: tuple[
        Annotated[HIPSDKCopyDomainReason, Field(strict=False)], ...
    ] = Field(default=(), strict=False)
    qualified: bool

    @model_validator(mode="after")
    def qualified_has_no_gaps(self) -> HIPSDKCopyDomain:
        """Qualified requires every in-scope copy lane-bound with no reasons."""
        if self.qualified and (
            self.reason_codes
            or any(c.in_scope and c.queue_id is None for c in self.copies)
        ):
            raise ValueError("qualified copy domain has unresolved lanes")
        if not self.qualified and not self.reason_codes:
            raise ValueError("unqualified copy domain requires reasons")
        return self


class HIPSDKScheduleReason(StrEnum):
    """Why a controlled-eager schedule is not complete."""

    COPY_DOMAIN = "copy_domain_unqualified"
    DEFAULT_STREAM = "default_stream_implicit_sync_not_inferred"
    CROSS_LANE_UNVERIFIED = "cross_lane_event_dependencies_unverified"


class HIPSDKScheduleQualification(BaseModelWithDocstrings):
    """Restricted-domain schedule completeness over audited eager submissions."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schedule_complete: bool
    default_stream_present: bool
    lane_count: int = Field(ge=0)
    reason_codes: tuple[
        Annotated[HIPSDKScheduleReason, Field(strict=False)], ...
    ] = Field(default=(), strict=False)

    @model_validator(mode="after")
    def complete_has_no_reasons(self) -> HIPSDKScheduleQualification:
        """Completeness requires an empty reason set; gaps stay explicit."""
        if self.schedule_complete and self.reason_codes:
            raise ValueError("complete schedule carries reasons")
        if not self.schedule_complete and not self.reason_codes:
            raise ValueError("incomplete schedule requires reasons")
        return self


class HIPSDKBoundDispatch(HIPBoundDispatch):
    """Preserve SDK stream zero for internal helpers; queue identity remains known.

    stream_id is the SDK ordinal used for lane co-observation. stream_handle
    is the raw hipStream_t argument; a zero handle is the legacy default
    stream and authorizes no sync inference.
    """

    stream_id: int = Field(ge=0)
    stream_handle: int | None = Field(default=None, ge=0)


class HIPTorchScope(CurrentFrozenSchemaModel):
    """Capture-local coverage proof with restricted-domain schedule completeness."""

    current_schema_version = SchemaVersion.HIP_TORCH_SCOPE
    schema_version: Literal[SchemaVersion.HIP_TORCH_SCOPE] = (
        SchemaVersion.HIP_TORCH_SCOPE
    )
    status: Literal["qualified_eager_scope"] = "qualified_eager_scope"
    scope: HIPTorchExpectation
    artifact_sha256: dict[str, str]
    process_id: int = Field(gt=0)
    dispatches: list[HIPSDKBoundDispatch]
    outside_dispatch_ids: list[int]
    implicit_dispatch_ids: list[int]
    links: list[HIPSDKEventLink]
    copy_domain: HIPSDKCopyDomain
    schedule: HIPSDKScheduleQualification
    performance_authority: Literal[False] = False
    schedule_complete: bool
    completeness_basis: Literal["sdk_enqueue_buffer_export_agreement"] = (
        "sdk_enqueue_buffer_export_agreement"
    )
    limitations: tuple[str, ...] = (
        "single_process_single_submission_thread_eager_only",
        "single_record_generation_per_event_in_scope",
        "shared_sdk_runtime_omissions_not_excluded",
        "schedule_limited_to_audited_eager_copy_and_event_domains",
        "not_timing_or_optimization_authority",
    )
