"""Neutral device admission; callers retain the native physical GPU lock.

This is a cooperative file protocol, not a scheduler or recovery authority.
It has no default record and never initializes a missing record to ready.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from sol_execbench.core.bench.gpu_lock_paths import (
    LEGACY_CLOCK_STATE_DIRECTORY,
    configured_gpu_locks,
)
from sol_execbench.core.control_plane_schema_versions import (
    ExecutionControlSchema,
)
from sol_execbench.core.data.base_model import (
    CurrentFrozenSchemaModel,
    FrozenArtifactModel,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value

ADMISSION_ENV = "SOL_EXECBENCH_ADMISSION_RECORD"
CLOCK_STATE_ENV = "SOL_EXECBENCH_CLOCK_STATE_DIR"
PROJECT = "sol"


class AdmissionError(RuntimeError):
    """Device admission or ownership has not been established."""


class AdmissionState(StrEnum):
    """Frozen neutral admission states."""

    READY = "ready"
    HELD = "held"
    BLOCKED = "blocked"


class AdmissionDevice(FrozenArtifactModel):
    """Canonical physical AMD identity, never a visible device index."""

    pci_bdf: str = Field(
        strict=True,
        pattern=r"^[0-9a-f]{4}:[0-9a-f]{2}:[01][0-9a-f]\.[0-7]$",
        min_length=12,
        max_length=12,
    )
    smi_uuid: str = Field(
        strict=True,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        min_length=36,
        max_length=36,
    )


class AdmissionOwner(FrozenArtifactModel):
    """Original project and native lease identity."""

    project: str = Field(strict=True, pattern=r"^[a-z][a-z0-9-]{0,63}$")
    lease_id: str = Field(
        strict=True, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$"
    )


class AdmissionRecord(CurrentFrozenSchemaModel):
    """The frozen six-field neutral record, with no project-private fields."""

    current_schema_version = ExecutionControlSchema.GPU_ADMISSION
    schema_version: Literal[ExecutionControlSchema.GPU_ADMISSION] = (
        ExecutionControlSchema.GPU_ADMISSION
    )
    device: AdmissionDevice
    state: AdmissionState
    owner: AdmissionOwner | None
    record_ref: Path | None
    reason: str | None = Field(strict=True)

    @model_validator(mode="after")
    def valid_state(self) -> Self:
        """Require the frozen state's owner, reference and reason semantics."""
        if self.record_ref is not None and (
            not self.record_ref.is_absolute()
            or len(str(self.record_ref)) < 2
            or "\0" in str(self.record_ref)
        ):
            raise ValueError("admission_record_ref_must_be_absolute")
        if self.state == AdmissionState.BLOCKED:
            if self.reason is None or not self.reason.strip():
                raise ValueError("admission_blocked_reason_required")
            if self.owner is not None and self.record_ref is None:
                raise ValueError("admission_owner_reference_required")
        elif self.record_ref is None or self.reason is not None:
            raise ValueError("admission_reference_or_reason_invalid")
        elif (self.owner is None) != (self.state == AdmissionState.READY):
            raise ValueError("admission_owner_state_invalid")
        return self


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("admission_duplicate_json_key")
        result[key] = value
    return result


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_durable_directory(path: Path) -> None:
    """Persist newly created private directories before publishing receipts."""
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _sync_directory(directory.parent)


def durable_json_write(path: Path, value: object) -> None:
    """Persist atomic JSON publication and its containing directory entry."""
    ensure_durable_directory(path.parent)
    atomic_write_json_value(path, value)
    _sync_directory(path.parent)


@dataclass(frozen=True, slots=True, kw_only=True)
class Admission:
    """Read and publish neutral state while the caller owns physical exclusion."""

    path: Path

    def read(self, device: AdmissionDevice) -> AdmissionRecord:
        """Reject unreadable, unknown, partial or differently targeted state."""
        try:
            value = json.loads(
                self.path.read_text(), object_pairs_hook=_unique_object
            )
            record = AdmissionRecord.model_validate(value)
        except (OSError, ValueError) as error:
            raise AdmissionError("gpu_admission_record_invalid") from error
        if record.device != device:
            raise AdmissionError("gpu_admission_device_mismatch")
        return record

    def require_ready(self, device: AdmissionDevice) -> None:
        """Observe readiness without reserving or modifying it."""
        if self.read(device).state != AdmissionState.READY:
            raise AdmissionError("gpu_admission_not_ready")

    def require_owner(
        self, device: AdmissionDevice, lease_id: str, reference: Path
    ) -> AdmissionRecord:
        """Require this SOL lease, including its original private record path."""
        record = self.read(device)
        if (
            record.state == AdmissionState.READY
            or record.owner
            != AdmissionOwner(project=PROJECT, lease_id=lease_id)
            or record.record_ref != reference
        ):
            raise AdmissionError("gpu_admission_foreign_or_unknown_owner")
        return record

    def hold(
        self, device: AdmissionDevice, lease_id: str, reference: Path
    ) -> None:
        """Publish held before the native responsibility record and side effects."""
        self.require_ready(device)
        record = AdmissionRecord(
            device=device,
            state=AdmissionState.HELD,
            owner=AdmissionOwner(project=PROJECT, lease_id=lease_id),
            record_ref=reference,
            reason=None,
        )
        durable_json_write(self.path, record.model_dump(mode="json"))

    def release(
        self, device: AdmissionDevice, lease_id: str, reference: Path
    ) -> None:
        """Publish ready only after the caller durably confirmed native cleanup."""
        record = self.require_owner(device, lease_id, reference)
        ready = record.model_copy(
            update={
                "state": AdmissionState.READY,
                "owner": None,
                "reason": None,
            }
        )
        try:
            durable_json_write(self.path, ready.model_dump(mode="json"))
        except OSError:
            # A rename may have succeeded before directory fsync failed.
            blocked = record.model_copy(
                update={
                    "state": AdmissionState.BLOCKED,
                    "reason": "ready_publication_failed",
                }
            )
            durable_json_write(self.path, blocked.model_dump(mode="json"))
            raise


def clock_state_directory(fallback: Path) -> Path:
    """Keep native receipts private when an explicit directory is selected."""
    value = os.environ.get(CLOCK_STATE_ENV)
    if value is None:
        configured = os.environ.get("GPU_LOCK_DIR")
        if (
            configured is not None
            and fallback.resolve() == Path(configured).resolve()
        ):
            return LEGACY_CLOCK_STATE_DIRECTORY
        return fallback
    path = Path(value)
    if not value or not path.is_absolute():
        raise AdmissionError("clock_state_directory_must_be_absolute")
    return path


def configured_admission() -> Admission | None:
    """Opt in explicitly; neither installation nor a missing file opts in."""
    value = os.environ.get(ADMISSION_ENV)
    if value is None:
        return None
    path = Path(value)
    private = os.environ.get(CLOCK_STATE_ENV)
    shared = configured_gpu_locks().directory
    if not value or not path.is_absolute() or not private:
        raise AdmissionError("gpu_admission_explicit_paths_required")
    if clock_state_directory(shared).resolve() == shared.resolve():
        raise AdmissionError("gpu_admission_private_state_required")
    return Admission(path=path)
