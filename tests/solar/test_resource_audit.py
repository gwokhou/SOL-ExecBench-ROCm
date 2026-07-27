from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import pytest

from solar.rocm.architecture import (
    RESOURCE_PEAK_CALIBRATION_SCHEMA_VERSION,
    RESOURCE_PEAK_TIMING_PROFILE,
    resource_peak_payload_sha256,
    verify_resource_peak_audit,
)

_EXPECTED_PRECISIONS = ("fp16",)
_EXPECTED_RESOURCE_MODES = ("valu/fp16",)
_EXPECTED_INSTRUCTION_CHECKS = ("fp16_valu_fma",)


def _telemetry() -> dict:
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


def _telemetry_summary(raw_batches: list[dict]) -> dict:
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


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _measurement_evidence() -> dict:
    raw_values = (
        (88.0, 90.0),
        (89.0, 91.0),
        (90.0, 92.0),
        (89.5, 90.5),
        (89.8, 90.2),
    )
    raw_batches = [
        {
            "process_batch": process_batch,
            "samples": list(values),
            "median": statistics.median(values),
            "telemetry_before": _telemetry(),
            "telemetry_after": _telemetry(),
        }
        for process_batch, values in enumerate(raw_values)
    ]
    samples = [sample for values in raw_values for sample in values]
    batch_medians = [statistics.median(values) for values in raw_values]
    lower_quartile = _quantile(samples, 0.25)
    upper_quartile = _quantile(samples, 0.75)
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
        "telemetry_summary": _telemetry_summary(raw_batches),
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
                "lower": 89.0,
                "upper": 91.0,
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
        "measured_ops_per_second": 90.0,
        "best_observed_ops_per_second": 92.0,
        "nominal_ops_per_second": 100.0,
        "measured_to_nominal_ratio": 0.9,
        "best_observed_to_nominal_ratio": 0.92,
    }


def _tuning_candidate(
    candidate_value: int,
    batch_centers: tuple[float, ...],
    *,
    binary_sha256: str,
) -> dict:
    raw_values = tuple((center - 0.5, center + 0.5) for center in batch_centers)
    raw_batches = [
        {
            "process_batch": process_batch,
            "samples": list(values),
            "median": statistics.median(values),
        }
        for process_batch, values in enumerate(raw_values)
    ]
    samples = [sample for values in raw_values for sample in values]
    batch_medians = [statistics.median(values) for values in raw_values]
    return {
        "configuration": {"waves_per_reported_wgp": candidate_value},
        "compiler_defines": {"SOL_WMMA_WAVES_PER_WGP": candidate_value},
        "binary_sha256": binary_sha256,
        "process_batch_count": len(raw_batches),
        "sample_count": len(samples),
        "batch_medians": batch_medians,
        "selection_result": statistics.median(batch_medians),
        "all_sample_median": statistics.median(samples),
        "interquartile_range": (
            _quantile(samples, 0.75) - _quantile(samples, 0.25)
        ),
        "raw_process_batches": raw_batches,
    }


def _add_tuning_evidence(payload: dict) -> None:
    measurement = payload["measurements"][0]
    candidates = [
        _tuning_candidate(8, (80.0, 81.0, 82.0), binary_sha256="8" * 64),
        _tuning_candidate(16, (90.0, 91.0, 92.0), binary_sha256="b" * 64),
    ]
    measurement.update(
        binary_sha256="b" * 64,
        compiler_defines={"SOL_WMMA_WAVES_PER_WGP": 16},
        selected_configuration={"waves_per_reported_wgp": 16},
        tuning={
            "status": "performed",
            "phase": "configuration_selection_only",
            "search_method": "deterministic exhaustive search",
            "parameter": "waves_per_reported_wgp",
            "candidate_values": [8, 16],
            "execution_order": [
                {"tuning_round": 0, "waves_per_reported_wgp": 8},
                {"tuning_round": 0, "waves_per_reported_wgp": 16},
                {"tuning_round": 1, "waves_per_reported_wgp": 16},
                {"tuning_round": 1, "waves_per_reported_wgp": 8},
                {"tuning_round": 2, "waves_per_reported_wgp": 8},
                {"tuning_round": 2, "waves_per_reported_wgp": 16},
            ],
            "candidates": candidates,
            "selected_configuration": {"waves_per_reported_wgp": 16},
            "held_out_samples_used_for_selection": False,
        },
    )


def _payload() -> dict:
    payload = {
        "schema_version": RESOURCE_PEAK_CALIBRATION_SCHEMA_VERSION,
        "timing_profile": RESOURCE_PEAK_TIMING_PROFILE,
        "device": {
            "device_name": "AMD Radeon RX 9060 XT",
            "gfx_target": "gfx1200",
        },
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
                    "probe": "fp16.hip",
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
            "required_checks": ["fp16_valu_fma"],
            "checks": {
                "fp16_valu_fma": {
                    "expectation": "native",
                    "status": "passed",
                    "probe": "fp16.hip",
                    "instructions": ["V_PK_FMA_F16"],
                    "isa_declared": True,
                    "compiler_emitted_count": 4,
                    "runtime_probe_passed": True,
                    "native_instruction_usable": True,
                },
            },
        },
        "calibration_coverage": {
            "status": "passed",
            "required_precisions": ["fp16"],
            "covered_precisions": ["fp16"],
            "required_resource_modes": ["valu/fp16"],
            "covered_resource_modes": ["valu/fp16"],
        },
        "measurements": [
            {
                "probe": "fp16.hip",
                "covers_precisions": ["fp16"],
                "covers_resource_modes": ["valu/fp16"],
                "runtime_probe_passed": True,
                **_measurement_evidence(),
                "compiler_isa": {
                    "matched_instruction_counts": {"V_PK_FMA_F16": 4},
                    "spec_provenance": {"spec_sha256": "a" * 64},
                },
            },
        ],
    }
    payload["payload_sha256"] = resource_peak_payload_sha256(payload)
    return payload


def _write(path: Path, payload: dict, *, resign: bool = True) -> str:
    if resign:
        payload["payload_sha256"] = resource_peak_payload_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(
    path: Path,
    digest: str,
    *,
    expected_unthrottled: bool = True,
) -> dict:
    return verify_resource_peak_audit(
        path,
        expected_sha256=digest,
        expected_schema_version=RESOURCE_PEAK_CALIBRATION_SCHEMA_VERSION,
        expected_timing_profile=RESOURCE_PEAK_TIMING_PROFILE,
        expected_clocks_locked=True,
        expected_unthrottled=expected_unthrottled,
        expected_gfx_target="gfx1200",
        expected_precisions=_EXPECTED_PRECISIONS,
        expected_resource_modes=_EXPECTED_RESOURCE_MODES,
        expected_instruction_checks=_EXPECTED_INSTRUCTION_CHECKS,
    )


def test_verifies_content_addressed_locked_clock_audit(tmp_path: Path):
    path = tmp_path / "audit.json"
    digest = _write(path, _payload())

    verified = _verify(path, digest)

    assert verified["instruction_validation"]["status"] == "passed"


def test_limited_scope_accepts_explicitly_reported_throttling(tmp_path: Path):
    path = tmp_path / "audit.json"
    payload = _payload()
    measurement = payload["measurements"][0]
    for batch in measurement["raw_process_batches"]:
        batch["telemetry_before"]["throttle_status"] = "THROTTLED"
        batch["telemetry_after"]["throttle_status"] = "THROTTLED"
    measurement["telemetry_summary"]["throttle_statuses"] = ["THROTTLED"]
    digest = _write(path, payload)

    verified = _verify(path, digest, expected_unthrottled=False)

    assert verified["measurements"][0]["telemetry_summary"][
        "throttle_statuses"
    ] == [
        "THROTTLED",
    ]
    with pytest.raises(ValueError, match="telemetry reports throttling"):
        _verify(path, digest)


def test_rejects_file_and_payload_identity_mismatches(tmp_path: Path):
    path = tmp_path / "audit.json"
    payload = _payload()
    digest = _write(path, payload)

    with pytest.raises(ValueError, match="identity mismatch"):
        _verify(path, "0" * 64)

    payload["device"]["device_name"] = "tampered"
    tampered_digest = _write(path, payload, resign=False)
    with pytest.raises(ValueError, match="payload checksum mismatch"):
        _verify(path, tampered_digest)

    assert digest != tampered_digest


def test_verifies_fallback_instruction_evidence_from_measurement(
    tmp_path: Path,
):
    path = tmp_path / "fallback.json"
    payload = _payload()
    check = payload["instruction_validation"]["checks"]["fp16_valu_fma"]
    check.update(
        expectation="fallback",
        instructions=["V_PK_FMA_BF16"],
        fallback_instructions=["V_FMAAK_F32"],
        isa_declared=False,
        compiler_emitted_count=0,
        fallback_emitted_count=8,
        native_instruction_usable=False,
    )
    payload["measurements"][0]["compiler_isa"]["matched_instruction_counts"] = {
        "V_PK_FMA_BF16": 0,
        "V_FMAAK_F32": 8,
    }
    payload["isa_spec_evidence"]["instruction_presence"] = {
        "V_PK_FMA_BF16": False,
        "V_FMAAK_F32": True,
    }

    digest = _write(path, payload)
    _verify(path, digest)

    check["fallback_emitted_count"] = 7
    digest = _write(path, payload)
    with pytest.raises(
        ValueError,
        match="fallback instruction evidence mismatch",
    ):
        _verify(path, digest)


def test_verifies_tuning_selection_is_separate_and_recomputable(tmp_path: Path):
    path = tmp_path / "tuned.json"
    payload = _payload()
    _add_tuning_evidence(payload)

    digest = _write(path, payload)
    verified = _verify(path, digest)

    assert verified["measurements"][0]["selected_configuration"] == {
        "waves_per_reported_wgp": 16,
    }

    payload["measurements"][0]["tuning"][
        "held_out_samples_used_for_selection"
    ] = True
    digest = _write(path, payload)
    with pytest.raises(ValueError, match="tuning protocol is invalid"):
        _verify(path, digest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda item: item["measurements"][0]["raw_process_batches"][0][
                "samples"
            ].__setitem__(0, 87.0),
            "batch median mismatch",
        ),
        (
            lambda item: item["measurements"][0]["statistics"][
                "bootstrap_median_confidence_interval_95"
            ].update(seed=999),
            "confidence interval is invalid",
        ),
        (
            lambda item: item["experiment_protocol"][
                "held_out_execution_order"
            ][0].update(position=1),
            "execution order is invalid",
        ),
        (
            lambda item: item["measurements"][0]["raw_process_batches"][0][
                "telemetry_before"
            ].update(deep_sleep="ENABLED"),
            "telemetry state is incomplete",
        ),
        (
            lambda item: item["measurements"][0]["raw_process_batches"][0][
                "telemetry_before"
            ].update(throttle_status="THROTTLED"),
            "telemetry reports throttling",
        ),
        (
            lambda item: item["measurements"][0]["telemetry_summary"].update(
                snapshot_count=1,
            ),
            "telemetry summary count mismatch",
        ),
    ],
)
def test_rejects_unreproducible_measurement_evidence(
    tmp_path: Path,
    mutation,
    message,
):
    path = tmp_path / "audit.json"
    payload = _payload()
    mutation(payload)
    digest = _write(path, payload)

    with pytest.raises(ValueError, match=message):
        _verify(path, digest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda item: item.update(schema_version="unsupported"),
            "schema mismatch",
        ),
        (
            lambda item: item.update(timing_profile="diagnostic"),
            "timing profile mismatch",
        ),
        (
            lambda item: item["device"].update(gfx_target="gfx942"),
            "gfx target mismatch",
        ),
        (
            lambda item: item["clock_setup"].update(
                clock_locked_verified=False,
            ),
            "clock-lock state mismatch",
        ),
        (
            lambda item: item["instruction_validation"].update(status="failed"),
            "instruction validation did not pass",
        ),
        (
            lambda item: item["calibration_coverage"].update(
                covered_precisions=[],
            ),
            "covered_precisions mismatch",
        ),
        (
            lambda item: item["calibration_coverage"].update(
                covered_resource_modes=[],
            ),
            "covered_resource_modes mismatch",
        ),
        (
            lambda item: item["measurements"][0].update(covers_precisions=[]),
            "covered_precisions mismatch",
        ),
        (
            lambda item: item["measurements"][0].update(
                covers_resource_modes=[],
            ),
            "covered_resource_modes mismatch",
        ),
        (
            lambda item: item["instruction_validation"].update(
                required_checks=[],
            ),
            "required instruction checks mismatch",
        ),
        (
            lambda item: item["instruction_validation"]["checks"][
                "fp16_valu_fma"
            ].update(native_instruction_usable=False),
            "native instruction evidence mismatch",
        ),
        (
            lambda item: item["isa_spec_evidence"][
                "instruction_presence"
            ].update(
                V_PK_FMA_F16=False,
            ),
            "ISA declaration mismatch",
        ),
        (
            lambda item: item["instruction_validation"]["checks"][
                "fp16_valu_fma"
            ].update(compiler_emitted_count=3),
            "compiler instruction count mismatch",
        ),
        (
            lambda item: item["measurements"][0]["compiler_isa"][
                "spec_provenance"
            ].update(spec_sha256="b" * 64),
            "compiler ISA specification mismatch",
        ),
        (
            lambda item: item["measurements"][0].update(
                runtime_probe_passed=False,
            ),
            "runtime probe did not pass",
        ),
        (
            lambda item: item["measurements"][0].update(
                nominal_ops_per_second=80.0,
                measured_to_nominal_ratio=1.125,
                best_observed_to_nominal_ratio=1.15,
            ),
            "measurement exceeds nominal",
        ),
    ],
)
def test_rejects_invalid_audit_contract(tmp_path: Path, mutation, message):
    path = tmp_path / "audit.json"
    payload = _payload()
    mutation(payload)
    digest = _write(path, payload)

    with pytest.raises(ValueError, match=message):
        _verify(path, digest)
