# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Status aggregation helpers for static kernel evidence extractors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class StaticToolRunLike(Protocol):
    """Minimal tool-run fields consumed by generic aggregators."""

    @property
    def command(self) -> list[str]:
        """Return the executed command."""
        ...

    @property
    def status(self) -> object:
        """Return the domain status value."""
        ...

    @property
    def reason_code(self) -> object:
        """Return the stable domain reason code."""
        ...


@dataclass(frozen=True, slots=True)
class ExtractorStatusVocabulary[StatusT]:
    """Domain status values used by the generic extractor aggregator."""

    collected: StatusT
    partial: StatusT
    failed: StatusT
    unavailable: StatusT


@dataclass(frozen=True, slots=True)
class ExtractorReasonVocabulary[StatusT, ReasonT]:
    """Status-to-reason vocabulary for one extractor evidence schema."""

    collected_status: StatusT
    partial_status: StatusT
    failed_status: StatusT
    collected_reason: ReasonT
    partial_reason: ReasonT
    partial_disassembly_reason: ReasonT
    failed_reason: ReasonT
    timeout_reason: ReasonT
    unavailable_reason: ReasonT


def aggregate_extractor_status_value[StatusT](
    tool_runs: Sequence[StaticToolRunLike],
    vocabulary: ExtractorStatusVocabulary[StatusT],
) -> StatusT:
    """Return aggregate extractor status from individual tool runs."""
    executable_runs = [run for run in tool_runs if run.command]
    successes = [
        run for run in executable_runs if run.status == vocabulary.collected
    ]
    failures = [
        run for run in executable_runs if run.status == vocabulary.failed
    ]
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


def aggregate_extractor_reason_value[StatusT, ReasonT](
    tool_runs: Sequence[StaticToolRunLike],
    status: StatusT,
    vocabulary: ExtractorReasonVocabulary[StatusT, ReasonT],
) -> ReasonT:
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
        if any(
            run.reason_code == vocabulary.timeout_reason for run in tool_runs
        ):
            return vocabulary.timeout_reason
        return vocabulary.failed_reason
    return vocabulary.unavailable_reason
