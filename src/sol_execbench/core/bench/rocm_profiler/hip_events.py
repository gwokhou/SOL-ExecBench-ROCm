# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic HIP barrier records; timestamp order alone is never causal."""

from collections import defaultdict
from enum import StrEnum

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.data.base_model import BaseModelWithDocstrings


class HIPEventOperation(StrEnum):
    """SDK barrier operation names, distinct from host HIP API calls."""

    RECORD = "HIP_EVENT_RECORD"
    WAIT = "HIP_EVENT_WAIT"


class HIPEventRecord(BaseModelWithDocstrings):
    """One GPU barrier, scoped to its capture, process and device."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capture: str = Field(min_length=1)
    process_id: int = Field(ge=0)
    agent_id: int = Field(ge=0)
    queue_id: int = Field(ge=0)
    stream_id: int | None = Field(default=None, ge=0)
    event_handle: int = Field(ge=0)
    source_queue_id: int = Field(ge=0)
    operation: HIPEventOperation = Field(strict=False)
    start_ns: int = Field(ge=0)
    end_ns: int = Field(ge=0)
    generation: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid_interval(self) -> HIPEventRecord:
        """Reject malformed intervals before any dependency reasoning."""
        if self.end_ns < self.start_ns:
            raise ValueError("HIP event ends before it starts")
        if self.operation is HIPEventOperation.RECORD and (
            self.source_queue_id != self.queue_id
        ):
            raise ValueError("record source queue differs from record queue")
        return self


def normalize_event_generations(
    records: list[HIPEventRecord],
) -> tuple[list[HIPEventRecord], list[str]]:
    """Assign observed record ordinals; reused-handle waits remain unknown.

    GPU completion order cannot identify which host record call a wait consumed.
    A single observed record also does not prove a capture contains every record.
    Consequently ordinals are observations, never dispatch dependency authority.
    """
    groups: dict[tuple[str, int, int, int], list[HIPEventRecord]] = defaultdict(
        list
    )
    for record in records:
        key = (
            record.capture,
            record.process_id,
            record.agent_id,
            record.event_handle,
        )
        groups[key].append(record)
    result: list[HIPEventRecord] = []
    warnings: list[str] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: (item.end_ns, item.start_ns))
        recorded = [
            item
            for item in ordered
            if item.operation is HIPEventOperation.RECORD
        ]
        unique = len(
            {(item.start_ns, item.end_ns) for item in recorded}
        ) == len(recorded)
        generation = 0
        for record in ordered:
            item = record
            if item.operation is HIPEventOperation.RECORD and unique:
                generation += 1
                item = item.model_copy(update={"generation": generation})
            result.append(item)
        if not unique:
            warnings.append("hip_event_duplicate_or_ambiguous_record")
        if any(item.operation is HIPEventOperation.WAIT for item in group):
            warnings.append("hip_event_wait_generation_unverified")
        if len(recorded) > 1:
            warnings.append("hip_event_handle_reused")
    return result, sorted(set(warnings))
