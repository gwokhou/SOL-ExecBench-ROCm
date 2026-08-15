from __future__ import annotations

import hashlib
import json
import statistics
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from solar.rocm import architecture
from solar.rocm.architecture import (
    ArchitectureProfile,
    MemoryLevel,
    resource_peak_payload_sha256,
)
from solar.schema_versions import (
    ResourceModelVersion as SolarResourceModelVersion,
    SchemaVersion as SolarSchemaVersion,
)

_RESOURCES = {
    "mfma",
    "valu",
    "sfu",
    "reduction",
    "atomic",
    "scan_sort",
    "conversion",
}


def _audit_telemetry() -> dict:
    return {
        "captured_at": "2026-07-25T00:00:00+00:00",
        "gfx_clock_mhz": 2780.0,
        "gfx_max_clock_mhz": 2780.0,
        "memory_clock_mhz": 1258.0,
        "edge_temperature_c": 55.0,
        "hotspot_temperature_c": 65.0,
        "memory_temperature_c": 70.0,
        "socket_power_w": 100.0,
        "deep_sleep": "DISABLED",
        "throttle_status": "UNTHROTTLED",
        "performance_level": "STABLE_PEAK",
    }


def _audit_telemetry_summary(raw_batches: list[dict]) -> dict:
    snapshots = [
        batch[field_name]
        for batch in raw_batches
        for field_name in ("telemetry_before", "telemetry_after")
    ]
    numeric_fields = (
        "gfx_clock_mhz",
        "gfx_max_clock_mhz",
        "memory_clock_mhz",
        "edge_temperature_c",
        "hotspot_temperature_c",
        "memory_temperature_c",
        "socket_power_w",
    )
    return {
        "snapshot_count": len(snapshots),
        "deep_sleep_states": sorted({item["deep_sleep"] for item in snapshots}),
        "throttle_statuses": sorted(
            {item["throttle_status"] for item in snapshots},
        ),
        "performance_levels": sorted(
            {item["performance_level"] for item in snapshots},
        ),
        "numeric": {
            field_name: {
                "minimum": min(item[field_name] for item in snapshots),
                "median": statistics.median(
                    item[field_name] for item in snapshots
                ),
                "maximum": max(item[field_name] for item in snapshots),
            }
            for field_name in numeric_fields
        },
    }


def _audit_quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _held_out_audit_measurement() -> dict:
    raw_values = (
        (48.0, 50.0),
        (49.0, 51.0),
        (50.0, 52.0),
        (49.5, 50.5),
        (49.8, 50.2),
    )
    raw_batches = [
        {
            "process_batch": process_batch,
            "samples": list(values),
            "median": statistics.median(values),
            "telemetry_before": _audit_telemetry(),
            "telemetry_after": _audit_telemetry(),
        }
        for process_batch, values in enumerate(raw_values)
    ]
    samples = [sample for values in raw_values for sample in values]
    batch_medians = [statistics.median(values) for values in raw_values]
    lower_quartile = _audit_quantile(samples, 0.25)
    upper_quartile = _audit_quantile(samples, 0.75)
    mean = statistics.mean(samples)
    standard_deviation = statistics.stdev(samples)
    return {
        "measurement_phase": "held_out_after_configuration_freeze",
        "primary_statistic": "median_of_process_batch_medians",
        "result_scale": 1.0,
        "peak_result": max(samples),
        "median_result": statistics.median(batch_medians),
        "minimum_result": min(samples),
        "sample_count": len(samples),
        "process_batch_count": len(raw_batches),
        "samples_per_process_batch": 2,
        "raw_process_batches": raw_batches,
        "telemetry_summary": _audit_telemetry_summary(raw_batches),
        "statistics": {
            "primary_statistic": "median_of_process_batch_medians",
            "primary_result": statistics.median(batch_medians),
            "all_sample_median": statistics.median(samples),
            "all_sample_mean": mean,
            "all_sample_standard_deviation": standard_deviation,
            "coefficient_of_variation": standard_deviation / mean,
            "minimum": min(samples),
            "maximum": max(samples),
            "lower_quartile": lower_quartile,
            "upper_quartile": upper_quartile,
            "interquartile_range": upper_quartile - lower_quartile,
            "batch_medians": batch_medians,
            "bootstrap_median_confidence_interval_95": {
                "lower": 49.0,
                "upper": 51.0,
                "confidence_level": 0.95,
                "method": "percentile bootstrap over process-batch medians",
                "replicates": 10_000,
                "seed": 123,
            },
        },
        "compiler_defines": {},
        "selected_configuration": {},
        "tuning": {
            "status": "not_required",
            "selected_configuration": {},
        },
        "measured_ops_per_second": 50.0,
        "best_observed_ops_per_second": 52.0,
        "nominal_ops_per_second": 100.0,
        "measured_to_nominal_ratio": 0.5,
        "best_observed_to_nominal_ratio": 0.52,
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
        "resource_model_version": SolarResourceModelVersion.AMD.value,
        "resource_limits": {
            resource: {"generic": 10.0} for resource in _RESOURCES
        },
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
    assert profile.memory_hierarchy[0] == MemoryLevel(
        name="l1",
        scope="cu",
        capacity_bytes=32,
        bandwidth_bytes_per_second=50.0,
        source="spec",
    )
    assert profile.to_dict()["memory_hierarchy"][0]["name"] == "l1"

    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(_profile_data()), encoding="utf-8")
    from_file = ArchitectureProfile.load(path)
    assert from_file.source == str(path)
    packaged = ArchitectureProfile.load("RX_9060_XT")
    assert packaged.vendor == "AMD"
    assert packaged.profile_revision == "rx9060xt-amd-resource-v5"
    assert packaged.resource_rate_for("valu", "fp16") == 25_600_000_000_000.0
    assert packaged.resource_rate_for("valu", "generic") == 51_281_920_000_000.0
    assert "valu/generic" not in packaged.required_calibration_resource_modes()
    assert packaged.required_calibration_precisions() == (
        "bf16",
        "fp16",
        "fp32",
        "fp8",
        "int8",
    )
    packaged.require_verified_audit_evidence()
    with pytest.raises(FileNotFoundError, match="not found"):
        ArchitectureProfile.load("profile_that_does_not_exist")


def test_packaged_rx9060xt_audit_pins_corrected_probe_semantics():
    profile = ArchitectureProfile.load("RX_9060_XT")
    solar_root = Path(architecture.__file__).resolve().parents[1]
    repository_root = Path(architecture.__file__).resolve().parents[3]
    audit_path = solar_root / str(profile.audit_evidence["path"])
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    measurements = {item["probe"]: item for item in payload["measurements"]}

    assert (
        measurements["vector_fp16_fp16.hip"]["nominal_ops_per_second"]
        == 25_600_000_000_000.0
    )
    assert (
        measurements["vector_bf16_bf16.hip"]["nominal_ops_per_second"] is None
    )
    assert payload["instruction_validation"]["checks"]["fp32_valu_fma"][
        "instructions"
    ] == ["V_DUAL_FMAC_F32"]
    assert (
        payload["experiment_protocol"]["design"]
        == "two_phase_tuning_then_held_out_measurement"
    )
    assert payload["experiment_protocol"][
        "configuration_frozen_before_held_out_measurement"
    ]
    assert (
        payload["experiment_protocol"]["held_out_process_batches_per_probe"]
        == 7
    )
    assert payload["experiment_protocol"]["samples_per_process_batch"] == 7
    assert measurements["vector_fp32_fp32.hip"]["selected_configuration"] == {
        "accumulator_count": 16,
    }
    for probe in (
        "matrix_fp16_fp16_wmma.hip",
        "matrix_bf16_bf16_wmma.hip",
    ):
        assert measurements[probe]["selected_configuration"] == {
            "waves_per_reported_wgp": 128,
        }
    for probe in (
        "matrix_fp8_fp8_wmma.hip",
        "matrix_int8_int8_wmma.hip",
    ):
        assert measurements[probe]["selected_configuration"] == {
            "waves_per_reported_wgp": 32,
        }
    assert (
        "valu/generic"
        not in payload["calibration_coverage"]["required_resource_modes"]
    )

    probe_root = (
        repository_root
        / "src"
        / "sol_execbench"
        / "data"
        / "hardware_calibration_probes"
    )
    for probe, measurement in measurements.items():
        assert (
            measurement["source_sha256"]
            == hashlib.sha256((probe_root / probe).read_bytes()).hexdigest()
        )
    script_path = (
        repository_root
        / "scripts"
        / "internal"
        / "rdna4"
        / "run_rdna4_resource_peak_calibration.py"
    )
    assert (
        payload["calibration_script_sha256"]
        == hashlib.sha256(script_path.read_bytes()).hexdigest()
    )


def test_precision_and_resource_methods():
    profile = ArchitectureProfile.load(_profile_data())
    assert profile.normalize_precision("FLOAT16") == "fp16"
    assert profile.tensor_precision("torch.float8_e4m3fn", "fp32") == "fp8"
    assert profile.tensor_precision("torch.float16", "fp32") == "fp16"
    with pytest.raises(ValueError, match="not supported"):
        profile.tensor_precision("torch.float8_e5m2", "fp32")
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.update(name=""), "name is required"),
        (lambda item: item.update(vendor="NVI" + "DIA"), "AMD architecture"),
        (
            lambda item: item.update(memory_bandwidth_bytes_per_second=0),
            "bandwidth must be positive",
        ),
        (
            lambda item: item.update(peak_ops_per_second={"fp16": 0}),
            "positive peak",
        ),
        (
            lambda item: item.update(resource_model_version="old"),
            "resource_model_version",
        ),
        (
            lambda item: item["resource_limits"].pop("sfu"),
            "complete AMD resource set",
        ),
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
                unknown={"generic": "reason"},
            ),
            "unknown resource",
        ),
        (
            lambda item: item["calibration_exempt_modes"]["valu"].update(
                absent="reason",
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
            lambda item: item["precision_support"]["fp16"].update(
                calibration="maybe",
            ),
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
                audit_evidence={"status": "unavailable", "sha256": "BAD"},
            ),
            "lowercase SHA-256",
        ),
        (
            lambda item: item.update(audit_evidence={"status": "verified"}),
            "requires a SHA-256",
        ),
        (
            lambda item: item["memory_hierarchy"].append(
                {"name": "l1", "scope": "device", "capacity_bytes": 1},
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
                bandwidth_bytes_per_second=0,
            ),
            "bandwidths must be positive",
        ),
    ],
)
def test_profile_validation_rejects_incomplete_or_unsound_data(
    mutation,
    message,
):
    data = _profile_data()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        ArchitectureProfile.load(data)


def test_verified_audit_evidence_is_content_addressed(
    tmp_path: Path,
    monkeypatch,
):
    evidence = tmp_path / "evidence.json"
    data = _profile_data()
    required_resources = sorted(
        f"{resource}/{mode}"
        for resource, modes in data["resource_limits"].items()
        for mode in modes
        if mode not in data["calibration_exempt_modes"].get(resource, {})
    )
    payload = {
        "schema_version": SolarSchemaVersion.RESOURCE_PEAK_CALIBRATION.value,
        "timing_profile": "official",
        "device": {"device_name": "test gpu", "gfx_target": "gfx1200"},
        "clock_setup": {"clock_locked_verified": True},
        "experiment_protocol": {
            "design": "two_phase_tuning_then_held_out_measurement",
            "tuning_process_batches_per_candidate": 3,
            "held_out_process_batches_per_probe": 5,
            "samples_per_process_batch": 2,
            "held_out_execution_order": [
                {
                    "process_batch": process_batch,
                    "position": 0,
                    "probe": "test.hip",
                }
                for process_batch in range(5)
            ],
            "primary_statistic": "median_of_process_batch_medians",
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 123,
            "raw_samples_retained": True,
            "configuration_frozen_before_held_out_measurement": True,
        },
        "isa_spec_evidence": {
            "architecture": "gfx1200",
            "provenance": {
                "architecture": "gfx1200",
                "release": "test",
                "spec_sha256": "a" * 64,
            },
            "instruction_presence": {"V_PK_FMA_F16": True},
        },
        "instruction_validation": {
            "status": "passed",
            "required_checks": ["fp16_check"],
            "checks": {
                "fp16_check": {
                    "expectation": "native",
                    "status": "passed",
                    "probe": "test.hip",
                    "instructions": ["V_PK_FMA_F16"],
                    "isa_declared": True,
                    "compiler_emitted_count": 2,
                    "runtime_probe_passed": True,
                    "native_instruction_usable": True,
                },
            },
        },
        "calibration_coverage": {
            "status": "passed",
            "required_precisions": ["fp16"],
            "covered_precisions": ["fp16"],
            "required_resource_modes": required_resources,
            "covered_resource_modes": required_resources,
        },
        "measurements": [
            {
                "probe": "test.hip",
                "covers_precisions": ["fp16"],
                "covers_resource_modes": required_resources,
                "runtime_probe_passed": True,
                **_held_out_audit_measurement(),
                "compiler_isa": {
                    "matched_instruction_counts": {"V_PK_FMA_F16": 2},
                    "spec_provenance": {"spec_sha256": "a" * 64},
                },
            },
        ],
    }
    payload["payload_sha256"] = resource_peak_payload_sha256(payload)
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    profile_path = tmp_path / "profiles" / "arch" / "profile.yaml"
    profile_path.parent.mkdir(parents=True)
    monkeypatch.setattr(
        architecture,
        "_packaged_profile_path",
        lambda name: profile_path,
    )

    data["audit_evidence"] = {
        "status": "verified",
        "path": "evidence.json",
        "sha256": digest,
        "required_schema_version": (
            SolarSchemaVersion.RESOURCE_PEAK_CALIBRATION.value
        ),
        "required_timing_profile": "official",
        "required_clocks_locked": True,
        "required_unthrottled": True,
        "evidence_scope": architecture.UNTHROTTLED_RESOURCE_PEAK_SCOPE,
        "gfx_target": "gfx1200",
        "required_instruction_checks": ["fp16_check"],
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
            candidate,
            "audit_evidence",
            deepcopy(data["audit_evidence"]),
        )
        mutation(candidate.audit_evidence)
        with pytest.raises(ValueError, match=message):
            candidate.require_verified_audit_evidence()
