# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Small text normalization helpers shared across report and CLI code."""

from __future__ import annotations

import re
from collections.abc import Iterable

_ASCII_ALNUM = re.compile(r"[^a-z0-9]")


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


def ordered_unique[T](values: Iterable[T]) -> list[T]:
    """Return values with duplicates removed while preserving first-seen order."""
    return list(dict.fromkeys(values))


def normalize_ascii_alnum(value: str | None) -> str:
    """Lowercase text and retain only ASCII letters and decimal digits."""
    return _ASCII_ALNUM.sub("", (value or "").lower())
