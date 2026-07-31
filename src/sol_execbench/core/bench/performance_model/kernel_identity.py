# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical kernel-symbol identity shared by static and runtime evidence."""

from __future__ import annotations


def kernel_symbol_key(symbol: str) -> str | None:
    """Normalize runtime demangling and simple global Itanium kernel names."""
    value = symbol.removesuffix(".kd").split("(", maxsplit=1)[0]
    if not value.startswith("_Z"):
        return value or None
    index = 2
    while index < len(value) and value[index].isdigit():
        index += 1
    if index == 2:
        return None
    length = int(value[2:index])
    name = value[index : index + length]
    return name if len(name) == length else None


__all__ = ["kernel_symbol_key"]
