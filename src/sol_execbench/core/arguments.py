# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Argument-parser value converters shared by command modules."""

from __future__ import annotations

import argparse


def parse_bool(value: str) -> bool:
    """Convert common boolean spellings to bool for argparse."""

    normalized = value.lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def none_if_requested(value: str | None) -> str | None:
    """Normalize empty/none/null CLI values to ``None``."""

    if value is None:
        return None
    if value.lower() in {"", "none", "null"}:
        return None
    return value
