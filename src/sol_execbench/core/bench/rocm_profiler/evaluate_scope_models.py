# SPDX-License-Identifier: Apache-2.0
"""Controller-bound production scope artifacts; never performance authority."""

from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field

from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    ROCR_UUID_PATTERN,
    SHA256_PATTERN,
    UUID_PATTERN,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_models import (
    HIPSDKBoundDispatch,
    HIPSDKCopyDomain,
    HIPSDKEventLink,
    HIPSDKScheduleQualification,
    HIPSDKScheduleReason,
)
from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.integrity.schema_versions import SchemaVersion

SCOPE_LAUNCH_FILE = "scope-launch.json"
SCOPE_INTENT_FILE = "scope-intent.json"
SCOPE_RECEIPT_FILE = "scope-receipt.json"
SCOPE_LEDGER_FILE = "ledger.json"
SCOPE_TRACE_DIRECTORY = "trace"
SCOPE_TRACE_FILE = f"{SCOPE_TRACE_DIRECTORY}/scope_results.json"
SCOPE_RESULT_FILE = "qualification.json"
SCOPE_ENV = "SOL_EXECBENCH_SCOPE_REPLAY"
SCOPE_COLLECTOR_ENV = "SOL_SCOPE_COLLECTOR"
SCOPE_TOOL_ENV = "ROCP_TOOL_LIBRARIES"
SCOPE_LEDGER_ENV = "SOL_SCOPE_LEDGER"
SCOPE_RUN_ENV = "SOL_SCOPE_RUN"
SCOPE_SCHEMA_ENV = "SOL_SCOPE_SCHEMA"
SCOPE_MAX_ROWS_ENV = "SOL_SCOPE_MAX_ROWS"
SCOPE_ARTIFACTS = frozenset(
    {SCOPE_RECEIPT_FILE, SCOPE_LEDGER_FILE, SCOPE_TRACE_FILE}
)
SCOPE_ARCHIVE_SUFFIX = ".hip-scope"


class EvaluateScopeStatus(StrEnum):
    """Closed admission outcomes for production scope evidence."""

    QUALIFIED = "qualified"
    REJECTED = "rejected"


class EvaluateScopeReason(StrEnum):
    """Stable admission failures outside the SDK coverage checks."""

    CANONICAL = "scope_canonical_evaluation_not_passed"
    UNSUPPORTED = "scope_execution_kind_unsupported"
    COLLECTION = "scope_collection_failed"
    INVALID = "scope_evidence_invalid"


class EvaluateScopeIntent(CurrentFrozenSchemaModel):
    """Written by the parent before the candidate process can execute."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    current_schema_version = SchemaVersion.HIP_EVALUATE_INTENT
    schema_version: Literal[SchemaVersion.HIP_EVALUATE_INTENT] = (
        SchemaVersion.HIP_EVALUATE_INTENT
    )
    run_id: str = Field(pattern=UUID_PATTERN)
    workload_uuid: str
    workload_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_sha256: str = Field(pattern=SHA256_PATTERN)
    definition_sha256: str = Field(pattern=SHA256_PATTERN)
    device_uuid: str = Field(pattern=ROCR_UUID_PATTERN)
    application_command: tuple[str, ...] = Field(strict=False)
    runtime_sha256: dict[str, str]
    staged_sha256: dict[str, str]
    collector_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class EvaluateScopeReceipt(CurrentFrozenSchemaModel):
    """Candidate-process receipt must agree with the retained controller intent."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    current_schema_version = SchemaVersion.HIP_EVALUATE_RECEIPT
    schema_version: Literal[SchemaVersion.HIP_EVALUATE_RECEIPT] = (
        SchemaVersion.HIP_EVALUATE_RECEIPT
    )
    intent_sha256: str = Field(pattern=SHA256_PATTERN)
    input_sha256: str = Field(pattern=SHA256_PATTERN)
    process_id: int = Field(gt=0)
    device_uuid: str
    correct: Literal[True]
    completed: Literal[True]


class EvaluateScopeResult(CurrentFrozenSchemaModel):
    """Coverage qualification is conditional on the existing execution trust model."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    current_schema_version = SchemaVersion.HIP_EVALUATE_SCOPE
    schema_version: Literal[SchemaVersion.HIP_EVALUATE_SCOPE] = (
        SchemaVersion.HIP_EVALUATE_SCOPE
    )
    intent: EvaluateScopeIntent
    status: EvaluateScopeStatus
    reason: str | None = None
    artifact_sha256: dict[str, str] = Field(default_factory=dict)
    dispatches: list[HIPSDKBoundDispatch] = Field(default_factory=list)
    implicit_dispatch_ids: list[int] = Field(default_factory=list)
    outside_dispatch_ids: list[int] = Field(default_factory=list)
    links: list[HIPSDKEventLink] = Field(default_factory=list)
    copy_domain: HIPSDKCopyDomain = Field(
        default_factory=lambda: HIPSDKCopyDomain(qualified=True)
    )
    schedule: HIPSDKScheduleQualification = Field(
        default_factory=lambda: HIPSDKScheduleQualification(
            schedule_complete=False,
            default_stream_present=False,
            lane_count=0,
            reason_codes=(HIPSDKScheduleReason.CROSS_LANE_UNVERIFIED,),
        )
    )
    performance_authority: Literal[False] = False
    schedule_complete: bool = False
    adversarial_attestation: Literal[False] = False
    limitations: tuple[str, ...] = (
        "single_process_single_submission_thread_eager_only",
        "shared_sdk_runtime_omissions_not_excluded",
        "candidate_process_receipt_is_not_independent_security_attestation",
        "schedule_limited_to_audited_eager_copy_and_event_domains",
        "not_timing_or_optimization_authority",
    )
