# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Small validation primitives shared by ROCm audit readers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast


def audit_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"architecture audit evidence {field} must be an object")
    return cast(Mapping[str, object], value)


def audit_string_set(value: object, field: str) -> set[str]:
    if not isinstance(value, list):
        raise ValueError(f"architecture audit evidence {field} must be a list")
    strings = [item for item in value if isinstance(item, str) and item.strip()]
    if len(strings) != len(value) or len(set(strings)) != len(strings):
        raise ValueError(
            f"architecture audit evidence {field} must contain unique strings"
        )
    return set(strings)


def audit_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"architecture audit evidence {field} must be a nonnegative integer"
        )
    return value


def audit_sha256(value: object, field: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(
            f"architecture audit evidence {field} must be a lowercase SHA-256"
        )
    return digest
