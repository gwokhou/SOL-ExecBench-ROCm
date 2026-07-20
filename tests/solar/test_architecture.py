from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from solar.analysis.resources import RESOURCE_MODEL_VERSION
from solar.rocm import architecture
from solar.rocm.architecture import ArchitectureProfile, MemoryLevel


_RESOURCES = {
    "mfma",
    "valu",
    "sfu",
    "reduction",
    "atomic",
    "scan_sort",
    "conversion",
}


def _profile_data() -> dict:
    return {
        "name": "test_amd",
        "vendor": "AMD",
        "gfx_target": "gfx1200",
        "compute_units": 32,
        "memory_capacity_bytes": 1024,
        "memory_bandwidth_bytes_per_second": 100.0,
        "l2_bytes": 64,
        "last_level_cache_bytes": 128,
        "peak_ops_per_second": {"fp16": 100.0, "fp8": 200.0},
        "resource_model_version": RESOURCE_MODEL_VERSION,
        "resource_limits": {resource: {"generic": 10.0} for resource in _RESOURCES},
        "resource_limit_sources": {
            resource: f"source for {resource}" for resource in _RESOURCES
        },
        "calibration_exempt_modes": {"valu": {"generic": "analytical"}},
        "precision_support": {
            "fp16": {
                "hardware": "native",
                "software": "ROCm",
                "calibration": "required",
                "evidence": "measurement",
            },
            "fp8": {
                "hardware": "native",
                "software": "ROCm",
                "calibration": "exempt",
                "evidence": "published",
                "limitation": "no public calibration kernel",
            },
        },
        "profile_revision": "test-r1",
        "audit_evidence": {
            "status": "unavailable",
            "reason_code": "test_only",
        },
        "precision_aliases": {"float8_e4m3fn": "fp8"},
        "clock_hz": 2_000_000_000,
        "memory_hierarchy": [
            {
                "name": "l1",
                "scope": "cu",
                "capacity_bytes": 32,
                "bandwidth_bytes_per_second": 50,
                "source": "spec",
            },
            {"name": "vram", "scope": "device", "capacity_bytes": None},
        ],
    }


def test_memory_level_and_profile_load_normalize_all_fields(tmp_path: Path):
    unknown = MemoryLevel.load({"name": "vram", "scope": "device"})
    assert unknown.capacity_bytes is None
    assert unknown.bandwidth_bytes_per_second is None
    assert unknown.source is None

    profile = ArchitectureProfile.load(_profile_data())
    assert profile.name == "test_amd"
    assert profile.clock_hz == 2_000_000_000
    assert profile.memory_hierarchy[0] == MemoryLevel("l1", "cu", 32, 50.0, "spec")
    assert profile.cache_flush_bytes == 128
    assert profile.to_dict()["memory_hierarchy"][0]["name"] == "l1"

    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(_profile_data()), encoding="utf-8")
    from_file = ArchitectureProfile.load(path)
    assert from_file.source == str(path)
    packaged = ArchitectureProfile.load("RX_9060_XT")
    assert packaged.vendor == "AMD"
    with pytest.raises(FileNotFoundError, match="not found"):
        ArchitectureProfile.load("profile_that_does_not_exist")


def test_precision_resource_and_roofline_methods():
    profile = ArchitectureProfile.load(_profile_data())
    assert profile.normalize_precision("FLOAT16") == "fp16"
    assert profile.tensor_precision("torch.float8_e4m3fn", "fp32") == "fp8"
    assert profile.tensor_precision("torch.float16", "fp32") == "fp16"
    with pytest.raises(ValueError, match="not supported"):
        profile.tensor_precision("torch.float8_e5m2", "fp32")
    assert profile.peak_for("half") == 100.0
    with pytest.raises(ValueError, match="Precision"):
        profile.peak_for("fp64")

    assert profile.theoretical_seconds(200, 50, "fp16") == 2.0
    assert profile.theoretical_seconds_by_precision({"fp16": 50, "fp8": 100}, 50) == 2.0
    assert profile.resource_rate_for("VALU", "generic") == 10.0
    with pytest.raises(ValueError, match="Resource 'missing'"):
        profile.resource_rate_for("missing", "generic")

    no_generic = deepcopy(_profile_data())
    no_generic["resource_limits"]["valu"] = {"fp16": 12.0}
    no_generic["calibration_exempt_modes"] = {}
    specialized = ArchitectureProfile.load(no_generic)
    assert specialized.resource_rate_for("valu", "fp16") == 12.0
    with pytest.raises(ValueError, match="Resource mode"):
        specialized.resource_rate_for("valu", "fp32")

    work = {"valu": {"generic": 20}, "mfma": {"generic": 10}}
    assert profile.resource_seconds(work) == {"valu": 2.0, "mfma": 1.0}
    assert profile.theoretical_seconds_by_resources(work, fused_bytes=300) == 3.0
    assert profile.theoretical_seconds_by_resources({}, fused_bytes=50) == 0.5


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.update(name=""), "name is required"),
        (lambda item: item.update(vendor="NVI" + "DIA"), "AMD architecture"),
        (
            lambda item: item.update(memory_bandwidth_bytes_per_second=0),
            "bandwidth must be positive",
        ),
        (lambda item: item.update(peak_ops_per_second={"fp16": 0}), "positive peak"),
        (
            lambda item: item.update(resource_model_version="old"),
            "resource_model_version",
        ),
        (lambda item: item["resource_limits"].pop("sfu"), "complete AMD resource set"),
        (
            lambda item: item["resource_limits"]["sfu"].update(generic=0),
            "rates for sfu must be positive",
        ),
        (
            lambda item: item["resource_limit_sources"].pop("sfu"),
            "source is required for sfu",
        ),
        (
            lambda item: item["calibration_exempt_modes"].update(
                unknown={"generic": "reason"}
            ),
            "unknown resource",
        ),
        (
            lambda item: item["calibration_exempt_modes"]["valu"].update(
                absent="reason"
            ),
            "declared mode and reason",
        ),
        (
            lambda item: item["precision_support"].pop("fp8"),
            "describe every published peak",
        ),
        (
            lambda item: item["precision_support"]["fp16"].pop("evidence"),
            "must define",
        ),
        (
            lambda item: item["precision_support"]["fp16"].update(calibration="maybe"),
            "required or exempt",
        ),
        (
            lambda item: item["precision_support"]["fp16"].update(hardware=""),
            "fields must be non-empty",
        ),
        (
            lambda item: item["precision_support"]["fp8"].pop("limitation"),
            "requires a limitation",
        ),
        (lambda item: item.update(profile_revision=""), "profile_revision"),
        (
            lambda item: item.update(audit_evidence={"status": "unknown"}),
            "status must be",
        ),
        (
            lambda item: item.update(
                audit_evidence={"status": "unavailable", "sha256": "BAD"}
            ),
            "lowercase SHA-256",
        ),
        (
            lambda item: item.update(audit_evidence={"status": "verified"}),
            "requires a SHA-256",
        ),
        (
            lambda item: item["memory_hierarchy"].append(
                {"name": "l1", "scope": "device", "capacity_bytes": 1}
            ),
            "names must be unique",
        ),
        (
            lambda item: item["memory_hierarchy"][0].update(scope=""),
            "name and scope",
        ),
        (
            lambda item: item["memory_hierarchy"][0].update(capacity_bytes=0),
            "capacities must be positive",
        ),
        (
            lambda item: item["memory_hierarchy"][0].update(
                bandwidth_bytes_per_second=0
            ),
            "bandwidths must be positive",
        ),
    ],
)
def test_profile_validation_rejects_incomplete_or_unsound_data(mutation, message):
    data = _profile_data()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        ArchitectureProfile.load(data)


def test_verified_audit_evidence_is_content_addressed(tmp_path: Path, monkeypatch):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("verified", encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    profile_path = tmp_path / "profiles" / "arch" / "profile.yaml"
    profile_path.parent.mkdir(parents=True)
    monkeypatch.setattr(
        architecture, "_packaged_profile_path", lambda name: profile_path
    )

    data = _profile_data()
    data["audit_evidence"] = {
        "status": "verified",
        "path": "evidence.json",
        "sha256": digest,
    }
    profile = ArchitectureProfile.load(data)
    profile.require_verified_audit_evidence()

    unavailable = ArchitectureProfile.load(_profile_data())
    with pytest.raises(ValueError, match="test_only"):
        unavailable.require_verified_audit_evidence()
    for mutation, message in [
        (lambda item: item.update(path=""), "lacks a path"),
        (lambda item: item.update(path="missing"), "file is missing"),
        (lambda item: item.update(sha256="0" * 64), "identity mismatch"),
    ]:
        candidate = ArchitectureProfile.load(data)
        object.__setattr__(
            candidate, "audit_evidence", deepcopy(data["audit_evidence"])
        )
        mutation(candidate.audit_evidence)
        with pytest.raises(ValueError, match=message):
            candidate.require_verified_audit_evidence()
