# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from sol_execbench.core.bench.profile_summary.metric_values import (
    finite_number_or_none,
    normalize_metric_key,
)


def test_normalize_metric_key_preserves_unicode_alphanumeric_characters() -> (
    None
):
    assert normalize_metric_key("L2_Caché-Hit Rate") == "l2cachéhitrate"
    assert normalize_metric_key(None) == ""


def test_finite_number_or_none_rejects_non_finite_values() -> None:
    assert finite_number_or_none(3.5) == 3.5
    assert finite_number_or_none(float("nan")) is None
    assert finite_number_or_none(float("inf")) is None
