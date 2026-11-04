# SPDX-License-Identifier: Apache-2.0
"""Typed independent SDK ledger and bounded eager-workload qualification."""

from typing import Annotated, Literal

from pydantic import ConfigDict, Field

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
    """Synchronous HIP API exit and its return value."""

    kind: Literal["api"]
    scope: Literal[0, 1]
    corr: int = Field(gt=0)
    name: str = Field(min_length=1)
    result: int


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
            | SDKEventComplete,
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
    """SDK enqueue endpoints, including dispatches from internal HIP helpers."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    predecessor_dispatch_id: int = Field(gt=0)
    successor_dispatch_id: int = Field(gt=0)
    record_correlation_id: int = Field(gt=0)
    wait_correlation_id: int = Field(gt=0)
    event_handle: int = Field(gt=0)
    basis: Literal["sdk_enqueue_order"] = "sdk_enqueue_order"


class HIPSDKBoundDispatch(HIPBoundDispatch):
    """Preserve SDK stream zero for internal helpers; queue identity remains known.

    A zero stream value does not authorize a default-stream dependency inference.
    """

    stream_id: int = Field(ge=0)


class HIPTorchScope(CurrentFrozenSchemaModel):
    """Capture-local coverage proof, not a complete dependency schedule."""

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
    performance_authority: Literal[False] = False
    schedule_complete: Literal[False] = False
    completeness_basis: Literal["sdk_enqueue_buffer_export_agreement"] = (
        "sdk_enqueue_buffer_export_agreement"
    )
    limitations: tuple[str, ...] = (
        "single_process_single_submission_thread_eager_only",
        "single_record_generation_per_event_in_scope",
        "shared_sdk_runtime_omissions_not_excluded",
        "memory_and_other_dependency_domains_not_qualified",
        "not_timing_or_optimization_authority",
    )
