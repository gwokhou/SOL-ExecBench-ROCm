# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Small text normalization helpers shared across report and CLI code."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

_T = TypeVar("_T")


def markdown_table_cell(value: object) -> str:
    """Return text safe for a pipe-delimited Markdown table cell."""

    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def subprocess_text(value: str | bytes | None) -> str:
    """Normalize subprocess output to text."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def text_tail(value: object, *, limit: int = 4000) -> str:
    """Normalize *value* to text and return its last *limit* characters."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = str(value)
    return text[-limit:]


def ordered_unique(values: Iterable[_T]) -> list[_T]:
    """Return values with duplicates removed while preserving first-seen order."""

    return list(dict.fromkeys(values))
