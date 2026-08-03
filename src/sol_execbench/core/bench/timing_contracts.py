# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed callbacks shared by runtime and device timing layers."""

from __future__ import annotations

import typing
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimingCallbacks:
    """Per-invocation argument generation and post-run validation."""

    argument_provider: Callable[[], list[typing.Any]] | None = None
    validator: Callable[[list[typing.Any], typing.Any], None] | None = None


__all__ = ["TimingCallbacks"]
