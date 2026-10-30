from __future__ import annotations

import copy

import pytest

from solar.rocm.resource_peak_measurements import _verify_telemetry
from solar.rocm.throttle import (
    ThrottleStatus,
    classify_throttle,
    evidence_status,
)


def evidence(value=0):
    return {
        "raw": {
            "throttle_status": {"value": value, "type": "int"},
            "indep_throttle_status": {
                "value": 0 if value == 0 else 1 << 36,
                "type": "int",
            },
        },
        "metrics_header": {
            "format_revision": 1,
            "content_revision": 3,
            "structure_size": 120,
        },
        "api_version": {"major": 27, "minor": 1},
        "cli_version": "AMDSMI Tool: 27.1.0",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, "UNTHROTTLED"),
        (2, "THROTTLED"),
        (0xFFFFFFFF, "UNKNOWN"),
        (-1, "UNKNOWN"),
        (1 << 32, "UNKNOWN"),
        (True, "UNKNOWN"),
        ("0", "UNKNOWN"),
        ("N/A", "UNKNOWN"),
        (None, "UNKNOWN"),
    ],
)
def test_raw_field_classification(raw, expected):
    assert classify_throttle(raw) == expected


def test_raw_recomputation_does_not_trust_labels():
    item = evidence(0xFFFFFFFF)
    item["status"] = "UNTHROTTLED"
    assert evidence_status(item) == ThrottleStatus.UNKNOWN
    assert evidence_status(evidence(0)) == ThrottleStatus.UNTHROTTLED
    assert evidence_status(evidence(2)) == ThrottleStatus.THROTTLED
    item = evidence()
    item["raw"]["indep_throttle_status"]["value"] = 1 << 36
    assert evidence_status(item) == ThrottleStatus.UNKNOWN


@pytest.mark.parametrize(
    "field", ["raw", "metrics_header", "api_version", "cli_version"]
)
def test_missing_evidence_remains_unknown(field):
    item = evidence()
    item.pop(field)
    assert evidence_status(item) == ThrottleStatus.UNKNOWN


def snapshot():
    return {
        "captured_at": "2026-10-30T15:00:00Z",
        "gfx_clock_mhz": 2750,
        "gfx_max_clock_mhz": 2780,
        "memory_clock_mhz": 1258,
        "edge_temperature_c": 50,
        "hotspot_temperature_c": 60,
        "memory_temperature_c": 65,
        "socket_power_w": 100,
        "deep_sleep": "DISABLED",
        "performance_level": "STABLE_PEAK",
        "throttle_status": "UNTHROTTLED",
        "throttle_status_cli": "UNTHROTTLED",
        "throttle_evidence": evidence(),
    }


def test_legacy_string_cannot_prove_new_unthrottled_qualification():
    item = snapshot()
    item.pop("throttle_evidence")
    original = copy.deepcopy(item)
    with pytest.raises(ValueError, match="missing or unknown"):
        _verify_telemetry(item, require_unthrottled=True)
    _verify_telemetry(item, require_unthrottled=False)
    assert item == original


def test_unknown_is_not_accepted_as_no_throttle():
    item = snapshot()
    item["throttle_evidence"] = evidence(0xFFFFFFFF)
    item["throttle_status"] = "UNKNOWN"
    _verify_telemetry(item, require_unthrottled=False)
    with pytest.raises(ValueError, match="missing or unknown"):
        _verify_telemetry(item, require_unthrottled=True)
    item["throttle_status"] = "UNTHROTTLED"
    with pytest.raises(ValueError, match="disagrees"):
        _verify_telemetry(item, require_unthrottled=True)


def test_conflicting_cli_and_raw_observations_cannot_certify_unthrottled():
    item = snapshot()
    item["throttle_status_cli"] = "THROTTLED"
    item["throttle_status"] = "UNKNOWN"
    with pytest.raises(ValueError, match="missing or unknown"):
        _verify_telemetry(item, require_unthrottled=True)
