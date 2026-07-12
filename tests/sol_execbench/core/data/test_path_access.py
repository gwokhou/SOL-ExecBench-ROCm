from __future__ import annotations

from types import MappingProxyType

import pytest

from sol_execbench.core.data.path_access import (
    path_bool,
    path_dict,
    path_float_or_none,
    path_get,
    path_int,
    path_int_or_none,
    path_list,
    path_mapping_list,
    path_require,
    path_str_list,
    path_str_or_none,
)


def test_path_get_reads_nested_values_and_defaults() -> None:
    payload = {"a": {"b": {"c": 3}}}

    assert path_get(payload, "a.b.c") == 3
    assert path_get(payload, "a.x", default="missing") == "missing"


def test_path_get_accepts_mappings_and_stops_at_missing_or_scalar_segments() -> None:
    payload = MappingProxyType(
        {
            "nested": MappingProxyType({"value": None}),
            "scalar": 3,
        }
    )

    assert path_get(payload, "nested.value", default="missing") is None
    assert path_get(payload, "nested.missing.value", default="missing") == "missing"
    assert path_get(payload, "scalar.value", default="missing") == "missing"


def test_path_require_reports_source_and_path() -> None:
    with pytest.raises(ValueError, match="sidecar missing required field: a.b"):
        path_require({}, "a.b", source="sidecar")

    with pytest.raises(
        ValueError, match="payload missing required field: scalar.value"
    ):
        path_require({"scalar": 3}, "scalar.value")


def test_path_dict_accepts_only_dict_values() -> None:
    assert path_dict({"a": {"b": 1}}, "a") == {"b": 1}
    assert path_dict({"a": []}, "a") == {}
    assert path_dict({"a": []}, "a", default={"fallback": 1}) == {"fallback": 1}


def test_path_list_accepts_only_list_values() -> None:
    assert path_list({"items": [1, 2]}, "items") == [1, 2]
    assert path_list({"items": {}}, "items") == []


def test_scalar_coercion_helpers_are_conservative() -> None:
    payload = {
        "name": " demo ",
        "count": 3,
        "duration": 1.25,
        "bool_value": True,
    }

    assert path_str_or_none(payload, "name") == " demo "
    assert path_str_or_none(payload, "missing") is None
    assert path_int(payload, "count", default=0) == 3
    assert path_int(payload, "bool_value", default=7) == 7
    assert path_float_or_none(payload, "duration") == 1.25
    assert path_float_or_none(payload, "bool_value") is None


def test_optional_scalar_helpers_reject_bool_for_int() -> None:
    payload = {"count": 3, "enabled": True, "missing": None}

    assert path_int_or_none(payload, "count") == 3
    assert path_int_or_none(payload, "enabled") is None
    assert path_int_or_none(payload, "missing") is None
    assert path_bool(payload, "enabled") is True
    assert path_bool(payload, "count") is False


def test_list_helpers_filter_to_expected_shapes() -> None:
    payload = {
        "records": [{"id": "a"}, [], {"id": "b"}],
        "codes": ["ready", 3, "blocked"],
    }

    assert path_mapping_list(payload, "records") == [{"id": "a"}, {"id": "b"}]
    assert path_str_list(payload, "codes") == ["ready", "blocked"]
