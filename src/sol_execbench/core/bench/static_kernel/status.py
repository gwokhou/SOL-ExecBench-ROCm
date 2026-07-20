# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Status aggregation helpers for static kernel evidence extractors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar


_StatusT = TypeVar("_StatusT")
_ReasonT = TypeVar("_ReasonT")


class StaticToolRunLike(Protocol):
    @property
    def command(self) -> list[str]: ...

    @property
    def status(self) -> object: ...

    @property
    def reason_code(self) -> object: ...


@dataclass(frozen=True, slots=True)
class ExtractorStatusVocabulary(Generic[_StatusT]):
    """Domain status values used by the generic extractor aggregator."""

    collected: _StatusT
    partial: _StatusT
    failed: _StatusT
    unavailable: _StatusT


@dataclass(frozen=True, slots=True)
class ExtractorReasonVocabulary(Generic[_StatusT, _ReasonT]):
    """Status-to-reason vocabulary for one extractor evidence schema."""

    collected_status: _StatusT
    partial_status: _StatusT
    failed_status: _StatusT
    collected_reason: _ReasonT
    partial_reason: _ReasonT
    partial_disassembly_reason: _ReasonT
    failed_reason: _ReasonT
    timeout_reason: _ReasonT
    unavailable_reason: _ReasonT


def aggregate_extractor_status_value(
    tool_runs: Sequence[StaticToolRunLike],
    vocabulary: ExtractorStatusVocabulary[_StatusT],
) -> _StatusT:
    """Return aggregate extractor status from individual tool runs."""
    executable_runs = [run for run in tool_runs if run.command]
    successes = [run for run in executable_runs if run.status == vocabulary.collected]
    failures = [run for run in executable_runs if run.status == vocabulary.failed]
    if successes and len(successes) == len(executable_runs):
        return vocabulary.collected
    if successes:
        return vocabulary.partial
    if failures:
        # With no successes, a real extractor failure (command ran and failed)
        # must not be masked as "unavailable" when mixed with runs whose
        # toolchain was missing -- surface it as failed instead.
        return vocabulary.failed
    return vocabulary.unavailable


def aggregate_extractor_reason_value(
    tool_runs: Sequence[StaticToolRunLike],
    status: _StatusT,
    vocabulary: ExtractorReasonVocabulary[_StatusT, _ReasonT],
) -> _ReasonT:
    """Return aggregate extractor reason from aggregate status and tool runs."""
    if status == vocabulary.collected_status:
        return vocabulary.collected_reason
    if status == vocabulary.partial_status:
        if any(
            run.reason_code == vocabulary.partial_disassembly_reason
            for run in tool_runs
        ):
            return vocabulary.partial_disassembly_reason
        return vocabulary.partial_reason
    if status == vocabulary.failed_status:
        if any(run.reason_code == vocabulary.timeout_reason for run in tool_runs):
            return vocabulary.timeout_reason
        return vocabulary.failed_reason
    return vocabulary.unavailable_reason
