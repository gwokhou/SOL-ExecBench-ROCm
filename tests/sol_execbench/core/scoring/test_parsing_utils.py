# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

import pytest

from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence
from sol_execbench.core.scoring.parsing_utils import (
    ensure_dict,
    parse_confidence,
    parse_list,
    parse_optional_str,
    parse_str,
)


def test_parse_helpers_validate_json_shape() -> None:
    payload = {
        "name": "demo",
        "items": [1, 2],
        "optional": None,
        "confidence": "supported",
    }

    assert ensure_dict(payload, source="payload") is payload
    assert parse_str(payload, "name", source="payload") == "demo"
    assert parse_list(payload, "items", source="payload") == [1, 2]
    assert parse_optional_str(payload, "optional", source="payload") is None
    assert (
        parse_confidence(payload, "confidence", source="payload")
        == EstimateConfidence.SUPPORTED
    )


def test_parse_helpers_raise_source_qualified_errors() -> None:
    with pytest.raises(ValueError, match="payload must be an object"):
        ensure_dict([], source="payload")

    with pytest.raises(ValueError, match="payload.name must be a string"):
        parse_str({"name": 1}, "name", source="payload")

    with pytest.raises(ValueError, match="payload.confidence has invalid confidence"):
        parse_confidence({"confidence": "maybe"}, "confidence", source="payload")
