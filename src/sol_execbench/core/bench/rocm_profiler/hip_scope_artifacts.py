# SPDX-License-Identifier: Apache-2.0
"""Bounded reading and physical identity checks for controlled HIP scope."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.bench.rocm_profiler.hip_causality_json import (
    SDKCapture,
    SDKCausalDocument,
)
from sol_execbench.core.bench.rocm_profiler.hip_event_artifacts import (
    MAX_JSON_BYTES,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    HIPScopeExpectation,
    HIPScopeReason,
    HIPTraceJournal,
)
from sol_execbench.core.integrity.checksums import sha256_bytes
from sol_execbench.core.integrity.schema_versions import SchemaVersion


class HIPScopeError(ValueError):
    """Failure to qualify; no partially qualified artifact is returned."""

    def __init__(self, reason: HIPScopeReason) -> None:
        """Expose a stable code without application arguments."""
        super().__init__(reason.value)
        self.reason = reason


class _View(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class _Handle(_View):
    handle: int = Field(ge=0)


class _UUID(_View):
    bytes: dict[str, int]

    def hex_value(self) -> str:
        """Decode the qualified SDK little-endian 64-bit GPU ID as ROCR UUID."""
        if set(self.bytes) != {f"value{i}" for i in range(16)}:
            raise HIPScopeError(HIPScopeReason.IDENTITY)
        raw = bytes(self.bytes[f"value{i}"] for i in range(16))
        if any(raw[8:]):
            raise HIPScopeError(HIPScopeReason.IDENTITY)
        return f"GPU-{int.from_bytes(raw[:8], 'little'):016x}"


class _Agent(_View):
    id: _Handle
    uuid: _UUID


class _Symbol(_View):
    kernel_id: int = Field(ge=0)
    kernel_name: str


class HIPScopeCaptureView(_View):
    """Additional SDK fields needed to bind device and all kernel identities."""

    agents: list[_Agent]
    kernel_symbols: list[_Symbol]


def read_bound_json(
    path: Path, digest: str, *, max_bytes: int = MAX_JSON_BYTES
) -> bytes:
    """Require the controller-retained artifact digest before any parsing."""
    with path.open("rb") as handle:
        content = handle.read(max_bytes + 1)
    if len(content) > max_bytes or sha256_bytes(content) != digest:
        raise HIPScopeError(HIPScopeReason.INVALID)
    return content


def read_scope_capture(
    content: bytes,
) -> tuple[SDKCapture, HIPScopeCaptureView]:
    """Only one process capture is accepted, without slicing a merged trace."""
    payload = json.loads(content)
    document = SDKCausalDocument.model_validate(payload)
    if len(document.captures) != 1:
        raise HIPScopeError(HIPScopeReason.IDENTITY)
    return document.captures[0], HIPScopeCaptureView.model_validate(
        payload["rocprofiler-sdk-tool"][0]
    )


def scope_agent(
    capture: SDKCapture,
    view: HIPScopeCaptureView,
    journal: HIPTraceJournal,
    expected: HIPScopeExpectation,
) -> int:
    """Bind the pre-run scope to the journal and the actual process capture."""
    if (
        len(expected.application_command) != 6
        or expected.application_command[1] != expected.workload_id
        or expected.application_command[3:6]
        != (
            expected.run_id,
            expected.candidate_sha256,
            SchemaVersion.HIP_TRACE_JOURNAL,
        )
        or tuple(capture.metadata.command) != expected.application_command
        or capture.metadata.pid != journal.process_id
        or any(
            getattr(journal, key) != getattr(expected, key)
            for key in (
                "run_id",
                "workload_id",
                "candidate_sha256",
                "device_uuid",
            )
        )
    ):
        raise HIPScopeError(HIPScopeReason.IDENTITY)
    agents = [
        agent.id.handle
        for agent in view.agents
        if agent.uuid.hex_value() == expected.device_uuid
    ]
    if len(agents) != 1:
        raise HIPScopeError(HIPScopeReason.IDENTITY)
    return agents[0]
