# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Status aggregation helpers for static kernel evidence extractors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar


_StatusT = TypeVar("_StatusT")
_ReasonT = TypeVar("_ReasonT")


class StaticToolRunLike(Protocol):
    @property
    def command(self) -> list[str]: ...

    @property
    def status(self) -> object: ...

    @property
    def reason_code(self) -> object: ...


def aggregate_extractor_status_value(
    tool_runs: Sequence[StaticToolRunLike],
    *,
    collected: _StatusT,
    partial: _StatusT,
    failed: _StatusT,
    unavailable: _StatusT,
) -> _StatusT:
    """Return aggregate extractor status from individual tool runs."""
    executable_runs = [run for run in tool_runs if run.command]
    successes = [run for run in executable_runs if run.status == collected]
    failures = [run for run in executable_runs if run.status == failed]
    if successes and len(successes) == len(executable_runs):
        return collected
    if successes:
        return partial
    if failures:
        # With no successes, a real extractor failure (command ran and failed)
        # must not be masked as "unavailable" when mixed with runs whose
        # toolchain was missing -- surface it as failed instead.
        return failed
    return unavailable


def aggregate_extractor_reason_value(
    tool_runs: Sequence[StaticToolRunLike],
    *,
    status: object,
    collected_status: object,
    partial_status: object,
    failed_status: object,
    collected_reason: _ReasonT,
    partial_reason: _ReasonT,
    partial_disassembly_reason: _ReasonT,
    failed_reason: _ReasonT,
    timeout_reason: _ReasonT,
    unavailable_reason: _ReasonT,
) -> _ReasonT:
    """Return aggregate extractor reason from aggregate status and tool runs."""
    if status == collected_status:
        return collected_reason
    if status == partial_status:
        if any(run.reason_code == partial_disassembly_reason for run in tool_runs):
            return partial_disassembly_reason
        return partial_reason
    if status == failed_status:
        if any(run.reason_code == timeout_reason for run in tool_runs):
            return timeout_reason
        return failed_reason
    return unavailable_reason
