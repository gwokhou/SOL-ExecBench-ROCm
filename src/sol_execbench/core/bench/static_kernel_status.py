# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Status aggregation helpers for static kernel evidence extractors."""

from __future__ import annotations

from typing import Protocol


class StaticToolRunLike(Protocol):
    command: list[str]
    status: object
    reason_code: object


def aggregate_extractor_status_value(
    tool_runs: list[StaticToolRunLike] | tuple[StaticToolRunLike, ...],
    *,
    collected: object,
    partial: object,
    failed: object,
    unavailable: object,
) -> object:
    """Return aggregate extractor status from individual tool runs."""
    executable_runs = [run for run in tool_runs if run.command]
    successes = [run for run in executable_runs if run.status == collected]
    failures = [run for run in executable_runs if run.status == failed]
    if successes and len(successes) == len(executable_runs):
        return collected
    if successes:
        return partial
    if failures and len(failures) == len(executable_runs):
        return failed
    return unavailable


def aggregate_extractor_reason_value(
    tool_runs: list[StaticToolRunLike] | tuple[StaticToolRunLike, ...],
    *,
    status: object,
    collected_status: object,
    partial_status: object,
    failed_status: object,
    collected_reason: object,
    partial_reason: object,
    partial_disassembly_reason: object,
    failed_reason: object,
    timeout_reason: object,
    unavailable_reason: object,
) -> object:
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
