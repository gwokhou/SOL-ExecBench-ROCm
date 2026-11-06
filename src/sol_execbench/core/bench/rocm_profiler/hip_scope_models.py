# SPDX-License-Identifier: Apache-2.0
"""Contracts for an independently journaled, audited native HIP executor."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from sol_execbench.core.bench.rocm_profiler.hip_causality_models import (
    MAX_CAUSAL_RECORDS,
    HIPEventDispatchLink,
)
from sol_execbench.core.data.base_model import (
    BaseModelWithDocstrings,
    CurrentFrozenSchemaModel,
)
from sol_execbench.core.integrity.schema_versions import SchemaVersion

SHA256_PATTERN = r"^[0-9a-f]{64}$"
UUID_PATTERN = r"^[0-9a-f]{32}$"
ROCR_UUID_PATTERN = r"^GPU-[0-9a-f]{16}$"


class HIPJournalAPI(StrEnum):
    """Audited executor calls, not every possible HIP API."""

    LAUNCH = "hipLaunchKernel"
    RECORD = "hipEventRecord"
    WAIT = "hipStreamWaitEvent"
    DESTROY = "hipEventDestroy"
    SYNC = "hipDeviceSynchronize"
    EVENT_SYNC = "hipEventSynchronize"


class HIPScopeReason(StrEnum):
    """Failure of a required independently checked condition."""

    INVALID = "hip_scope_invalid_artifact"
    IDENTITY = "hip_scope_identity_mismatch"
    JOURNAL = "hip_scope_journal_mismatch"
    DISPATCH = "hip_scope_dispatch_mismatch"
    BARRIER = "hip_scope_barrier_mismatch"
    FENCE = "hip_scope_completion_fence_missing"
    CAUSALITY = "hip_scope_causality_unresolved"
    COPY = "copy_channel_mismatch"


class HIPScopeLimitation(StrEnum):
    """Bounds that remain after a controlled scope is qualified."""

    EXECUTOR = "audited_executor_and_call_domains_only"
    RUNTIME = "shared_runtime_bugs_not_excluded"
    DOMAINS = "memory_and_other_dependency_domains_not_qualified"
    AUTHORITY = "not_timing_or_optimization_authority"


class HIPJournalCall(BaseModelWithDocstrings):
    """Application-side observation written without reading profiler output."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: HIPJournalAPI = Field(strict=False)
    stream_handle: int = Field(ge=0)
    event_handle: int = Field(ge=0)
    kernel_name: str
    result: int


class HIPTraceJournal(CurrentFrozenSchemaModel):
    """A normally terminated independent journal from the trusted executor."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    current_schema_version = SchemaVersion.HIP_TRACE_JOURNAL
    schema_version: Literal[SchemaVersion.HIP_TRACE_JOURNAL] = (
        SchemaVersion.HIP_TRACE_JOURNAL
    )
    run_id: str = Field(pattern=UUID_PATTERN)
    workload_id: str = Field(min_length=1)
    candidate_sha256: str = Field(pattern=SHA256_PATTERN)
    device_uuid: str = Field(pattern=ROCR_UUID_PATTERN)
    process_id: int = Field(gt=0)
    thread_id: int = Field(gt=0)
    completed: Literal[True]
    correct: Literal[True]
    calls: list[HIPJournalCall] = Field(
        min_length=2, max_length=MAX_CAUSAL_RECORDS
    )


class HIPScopeExpectation(BaseModelWithDocstrings):
    """Controller-owned pre-run identity, never inferred from the trace.

    The caller must audit the executor's closed call set. Matching hashes binds
    that audit; it does not make an arbitrary candidate's own journal trusted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    run_id: str = Field(pattern=UUID_PATTERN)
    workload_id: str = Field(min_length=1)
    candidate_sha256: str = Field(pattern=SHA256_PATTERN)
    executor_source_sha256: str = Field(pattern=SHA256_PATTERN)
    device_uuid: str = Field(pattern=ROCR_UUID_PATTERN)
    application_command: tuple[str, ...] = Field(min_length=1, strict=False)
    returncode: Literal[0]
    executor: Literal["audited_native_hip_journal"] = (
        "audited_native_hip_journal"
    )


class HIPBoundDispatch(BaseModelWithDocstrings):
    """One member of the complete dispatch set of this controlled process."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    dispatch_id: int = Field(gt=0)
    correlation_id: int = Field(gt=0)
    kernel_name: str
    agent_id: int = Field(ge=0)
    queue_id: int = Field(ge=0)
    stream_id: int = Field(gt=0)


class HIPTraceScopeQualification(CurrentFrozenSchemaModel):
    """Qualified only relative to an independently audited executor call set."""

    current_schema_version = SchemaVersion.HIP_TRACE_SCOPE
    schema_version: Literal[SchemaVersion.HIP_TRACE_SCOPE] = (
        SchemaVersion.HIP_TRACE_SCOPE
    )
    status: Literal["qualified_controlled_scope"] = "qualified_controlled_scope"
    scope: HIPScopeExpectation
    trace_sha256: str = Field(pattern=SHA256_PATTERN)
    journal_sha256: str = Field(pattern=SHA256_PATTERN)
    process_id: int = Field(gt=0)
    dispatches: list[HIPBoundDispatch]
    links: list[HIPEventDispatchLink]
    journal_call_count: int = Field(gt=0)
    completeness_basis: Literal["independent_audited_executor_journal"] = (
        "independent_audited_executor_journal"
    )
    performance_authority: Literal[False] = False
    schedule_complete: Literal[False] = False
    limitations: list[Annotated[HIPScopeLimitation, Field(strict=False)]] = (
        Field(default_factory=lambda: list(HIPScopeLimitation))
    )
