# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public SOL-ExecBench dataset category contract."""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_CATEGORIES: tuple[str, ...] = ("FlashInfer-Bench", "L1", "L2", "Quant")
"""Canonical public benchmark categories in deterministic manifest order."""

_CATEGORY_SET = frozenset(DEFAULT_CATEGORIES)


def validate_categories(categories: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return validated categories in canonical deterministic order."""

    if categories is None:
        return DEFAULT_CATEGORIES

    selected = tuple(dict.fromkeys(categories))
    unknown = sorted(category for category in selected if category not in _CATEGORY_SET)
    if unknown:
        known = ", ".join(DEFAULT_CATEGORIES)
        invalid = ", ".join(unknown)
        raise ValueError(f"unknown SOL-ExecBench category: {invalid}; expected one of: {known}")

    selected_set = set(selected)
    return tuple(category for category in DEFAULT_CATEGORIES if category in selected_set)
