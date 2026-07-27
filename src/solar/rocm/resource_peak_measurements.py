# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Recompute and validate held-out resource-peak measurement evidence."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

from solar.rocm.audit_validation import (
    audit_mapping as _audit_mapping,
)
from solar.rocm.audit_validation import (
    audit_nonnegative_int as _audit_nonnegative_int,
)
from solar.rocm.audit_validation import (
    audit_sha256 as _audit_sha256,
)
from solar.rocm.audit_validation import (
    audit_string_set as _audit_string_set,
)

_NOMINAL_TOLERANCE = 1.001
_BOOTSTRAP_METHOD = "percentile bootstrap over process-batch medians"
_THROTTLE_STATUSES = frozenset({"THROTTLED", "UNTHROTTLED"})
_TELEMETRY_NUMERIC_FIELDS = (
    "gfx_clock_mhz",
    "gfx_max_clock_mhz",
    "memory_clock_mhz",
    "edge_temperature_c",
    "hotspot_temperature_c",
    "memory_temperature_c",
    "socket_power_w",
)


def verify_resource_peak_measurements(
    protocol_value: object,
    measurements_value: object,
    *,
    require_unthrottled: bool = True,
) -> tuple[set[str], set[str], dict[str, Mapping[str, object]]]:
    """Validate the experimental protocol and all raw measurement evidence."""
    (
        tuning_batches,
        held_out_batches,
        samples_per_batch,
        bootstrap_replicates,
        bootstrap_seed,
    ) = _verify_experiment_protocol(protocol_value)
    precisions, resources, measurements = _verify_measurements(
        measurements_value,
        expected_tuning_batches=tuning_batches,
        expected_held_out_batches=held_out_batches,
        expected_samples_per_batch=samples_per_batch,
        expected_bootstrap_replicates=bootstrap_replicates,
        expected_bootstrap_seed=bootstrap_seed,
        require_unthrottled=require_unthrottled,
    )
    _verify_execution_order(
        protocol_value,
        probes=set(measurements),
        expected_batches=held_out_batches,
    )
    return precisions, resources, measurements


def _verify_experiment_protocol(
    value: object,
) -> tuple[int, int, int, int, int]:
    protocol = _audit_mapping(value, "experiment_protocol")
    if protocol.get("design") != "two_phase_tuning_then_held_out_measurement":
        raise ValueError("architecture audit experiment design mismatch")
    if protocol.get("primary_statistic") != "median_of_process_batch_medians":
        raise ValueError("architecture audit primary statistic mismatch")
    if (
        protocol.get("raw_samples_retained") is not True
        or protocol.get("configuration_frozen_before_held_out_measurement")
        is not True
    ):
        raise ValueError(
            "architecture audit held-out measurement protocol is invalid",
        )
    tuning_batches = _positive_int(
        protocol.get("tuning_process_batches_per_candidate"),
        "tuning process batches",
    )
    held_out_batches = _positive_int(
        protocol.get("held_out_process_batches_per_probe"),
        "held-out process batches",
    )
    samples_per_batch = _positive_int(
        protocol.get("samples_per_process_batch"),
        "samples per process batch",
    )
    bootstrap_replicates = _positive_int(
        protocol.get("bootstrap_replicates"),
        "bootstrap replicates",
    )
    bootstrap_seed = protocol.get("bootstrap_seed")
    if (
        held_out_batches < 5
        or bootstrap_replicates < 1000
        or not isinstance(bootstrap_seed, int)
    ):
        raise ValueError(
            "architecture audit uncertainty protocol is incomplete",
        )
    return (
        tuning_batches,
        held_out_batches,
        samples_per_batch,
        bootstrap_replicates,
        bootstrap_seed,
    )


def _verify_execution_order(
    value: object,
    *,
    probes: set[str],
    expected_batches: int,
) -> None:
    protocol = _audit_mapping(value, "experiment_protocol")
    raw_order = protocol.get("held_out_execution_order")
    if not isinstance(raw_order, list):
        raise ValueError(
            "architecture audit held-out execution order is missing",
        )
    expected_count = len(probes) * expected_batches
    if len(raw_order) != expected_count:
        raise ValueError(
            "architecture audit held-out execution order is incomplete",
        )
    observed: set[tuple[int, str]] = set()
    observed_positions: set[tuple[int, int]] = set()
    for raw in raw_order:
        entry = _audit_mapping(raw, "held-out execution order entry")
        process_batch = _audit_nonnegative_int(
            entry.get("process_batch"),
            "process batch",
        )
        position = _audit_nonnegative_int(
            entry.get("position"),
            "probe position",
        )
        probe = str(entry.get("probe", ""))
        if (
            process_batch >= expected_batches
            or position >= len(probes)
            or probe not in probes
        ):
            raise ValueError(
                "architecture audit held-out execution order is invalid",
            )
        observed.add((process_batch, probe))
        observed_positions.add((process_batch, position))
    expected = {
        (process_batch, probe)
        for process_batch in range(expected_batches)
        for probe in probes
    }
    expected_positions = {
        (process_batch, position)
        for process_batch in range(expected_batches)
        for position in range(len(probes))
    }
    if observed != expected or observed_positions != expected_positions:
        raise ValueError(
            "architecture audit held-out execution order has duplicates",
        )


def _verify_measurements(
    value: object,
    *,
    expected_tuning_batches: int,
    expected_held_out_batches: int,
    expected_samples_per_batch: int,
    expected_bootstrap_replicates: int,
    expected_bootstrap_seed: int,
    require_unthrottled: bool,
) -> tuple[set[str], set[str], dict[str, Mapping[str, object]]]:
    if not isinstance(value, list) or not value:
        raise ValueError("architecture audit evidence has no measurements")
    covered_precisions: set[str] = set()
    covered_resources: set[str] = set()
    measurements: dict[str, Mapping[str, object]] = {}
    for raw in value:
        measurement = _audit_mapping(raw, "measurement")
        probe = str(measurement.get("probe", "")).strip()
        if not probe or probe in measurements:
            raise ValueError(
                "architecture audit measurement probe must be unique",
            )
        measurements[probe] = measurement
        covered_precisions.update(
            _audit_string_set(
                measurement.get("covers_precisions"),
                "covers_precisions",
            ),
        )
        covered_resources.update(
            _audit_string_set(
                measurement.get("covers_resource_modes"),
                "covers_resource_modes",
            ),
        )
        _verify_one_measurement(
            measurement,
            expected_tuning_batches=expected_tuning_batches,
            expected_held_out_batches=expected_held_out_batches,
            expected_samples_per_batch=expected_samples_per_batch,
            expected_bootstrap_replicates=expected_bootstrap_replicates,
            expected_bootstrap_seed=expected_bootstrap_seed,
            require_unthrottled=require_unthrottled,
        )
    return covered_precisions, covered_resources, measurements


def _verify_one_measurement(
    measurement: Mapping[str, object],
    *,
    expected_tuning_batches: int,
    expected_held_out_batches: int,
    expected_samples_per_batch: int,
    expected_bootstrap_replicates: int,
    expected_bootstrap_seed: int,
    require_unthrottled: bool,
) -> None:
    if measurement.get("runtime_probe_passed") is not True:
        raise ValueError("architecture audit runtime probe did not pass")
    _verify_held_out_measurement(
        measurement,
        expected_batches=expected_held_out_batches,
        expected_samples_per_batch=expected_samples_per_batch,
        expected_bootstrap_replicates=expected_bootstrap_replicates,
        expected_bootstrap_seed=expected_bootstrap_seed,
        require_unthrottled=require_unthrottled,
    )
    _verify_measurement_tuning(
        measurement,
        expected_tuning_batches=expected_tuning_batches,
    )
    nominal_raw = measurement.get("nominal_ops_per_second")
    if nominal_raw is None:
        return
    nominal = _positive_number(nominal_raw, "nominal rate")
    best_observed = _positive_number(
        measurement.get("best_observed_ops_per_second"),
        "best observed rate",
    )
    if best_observed > nominal * _NOMINAL_TOLERANCE:
        raise ValueError("architecture audit measurement exceeds nominal")


def _verify_held_out_measurement(
    measurement: Mapping[str, object],
    *,
    expected_batches: int,
    expected_samples_per_batch: int,
    expected_bootstrap_replicates: int,
    expected_bootstrap_seed: int,
    require_unthrottled: bool,
) -> None:
    if (
        measurement.get("measurement_phase")
        != "held_out_after_configuration_freeze"
        or measurement.get("primary_statistic")
        != "median_of_process_batch_medians"
    ):
        raise ValueError("architecture audit measurement is not held out")
    batches, samples, telemetry = _raw_batches(
        measurement.get("raw_process_batches"),
        expected_batches=expected_batches,
        expected_samples_per_batch=expected_samples_per_batch,
        require_telemetry=True,
        require_unthrottled=require_unthrottled,
    )
    _verify_telemetry_summary(measurement.get("telemetry_summary"), telemetry)
    if (
        measurement.get("process_batch_count") != expected_batches
        or measurement.get("samples_per_process_batch")
        != expected_samples_per_batch
        or measurement.get("sample_count") != len(samples)
    ):
        raise ValueError("architecture audit raw sample counts mismatch")
    batch_medians = tuple(statistics.median(batch) for batch in batches)
    primary = statistics.median(batch_medians)
    _close(measurement.get("median_result"), primary, "median result")
    _close(measurement.get("minimum_result"), min(samples), "minimum result")
    _close(measurement.get("peak_result"), max(samples), "peak result")
    _verify_statistics(
        _audit_mapping(measurement.get("statistics"), "measurement statistics"),
        samples=samples,
        batch_medians=batch_medians,
        expected_bootstrap_replicates=expected_bootstrap_replicates,
        expected_bootstrap_seed=expected_bootstrap_seed,
    )
    _verify_rates(measurement, samples=samples, primary=primary)


def _verify_rates(
    measurement: Mapping[str, object],
    *,
    samples: tuple[float, ...],
    primary: float,
) -> None:
    result_scale = _positive_number(
        measurement.get("result_scale"),
        "result scale",
    )
    measured = _positive_number(
        measurement.get("measured_ops_per_second"),
        "measured rate",
    )
    best = _positive_number(
        measurement.get("best_observed_ops_per_second"),
        "best observed rate",
    )
    _close(measured, primary * result_scale, "measured rate")
    _close(best, max(samples) * result_scale, "best observed rate")
    nominal_raw = measurement.get("nominal_ops_per_second")
    if nominal_raw is None:
        if (
            measurement.get("measured_to_nominal_ratio") is not None
            or measurement.get("best_observed_to_nominal_ratio") is not None
        ):
            raise ValueError(
                "architecture audit unpublished nominal has a ratio",
            )
        return
    nominal = _positive_number(nominal_raw, "nominal rate")
    _close(
        measurement.get("measured_to_nominal_ratio"),
        measured / nominal,
        "measured-to-nominal ratio",
    )
    _close(
        measurement.get("best_observed_to_nominal_ratio"),
        best / nominal,
        "best-observed-to-nominal ratio",
    )


def _raw_batches(
    value: object,
    *,
    expected_batches: int,
    expected_samples_per_batch: int,
    require_telemetry: bool,
    require_unthrottled: bool = False,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[Mapping[str, object], ...],
]:
    if not isinstance(value, list) or len(value) != expected_batches:
        raise ValueError("architecture audit raw process batches mismatch")
    batches: list[tuple[float, ...]] = []
    telemetry: list[Mapping[str, object]] = []
    for expected_index, raw in enumerate(value):
        batch = _audit_mapping(raw, "raw process batch")
        if batch.get("process_batch") != expected_index:
            raise ValueError("architecture audit process batch index mismatch")
        samples = _number_sequence(batch.get("samples"), "raw samples")
        if len(samples) != expected_samples_per_batch:
            raise ValueError("architecture audit samples-per-batch mismatch")
        _close(batch.get("median"), statistics.median(samples), "batch median")
        if require_telemetry:
            telemetry.append(
                _verify_telemetry(
                    batch.get("telemetry_before"),
                    require_unthrottled=require_unthrottled,
                ),
            )
            telemetry.append(
                _verify_telemetry(
                    batch.get("telemetry_after"),
                    require_unthrottled=require_unthrottled,
                ),
            )
        batches.append(samples)
    frozen_batches = tuple(batches)
    flattened = tuple(sample for batch in frozen_batches for sample in batch)
    return frozen_batches, flattened, tuple(telemetry)


def _verify_telemetry(
    value: object,
    *,
    require_unthrottled: bool,
) -> Mapping[str, object]:
    snapshot = _audit_mapping(value, "telemetry snapshot")
    if not str(snapshot.get("captured_at", "")).strip():
        raise ValueError("architecture audit telemetry timestamp is missing")
    for field_name in _TELEMETRY_NUMERIC_FIELDS:
        _positive_number(snapshot.get(field_name), f"telemetry {field_name}")
    deep_sleep = str(snapshot.get("deep_sleep", ""))
    performance_level = str(snapshot.get("performance_level", ""))
    throttle_status = str(snapshot.get("throttle_status", ""))
    if (
        deep_sleep != "DISABLED"
        or "STABLE_PEAK" not in performance_level
        or throttle_status not in _THROTTLE_STATUSES
    ):
        raise ValueError("architecture audit telemetry state is incomplete")
    if require_unthrottled and throttle_status != "UNTHROTTLED":
        raise ValueError("architecture audit telemetry reports throttling")
    current_clock = _positive_number(
        snapshot.get("gfx_clock_mhz"),
        "telemetry gfx clock",
    )
    maximum_clock = _positive_number(
        snapshot.get("gfx_max_clock_mhz"),
        "telemetry maximum gfx clock",
    )
    if current_clock < maximum_clock * 0.95:
        raise ValueError(
            "architecture audit telemetry clock is below locked range",
        )
    return snapshot


def _verify_telemetry_summary(
    value: object,
    snapshots: tuple[Mapping[str, object], ...],
) -> None:
    summary = _audit_mapping(value, "telemetry_summary")
    if summary.get("snapshot_count") != len(snapshots):
        raise ValueError("architecture audit telemetry summary count mismatch")
    expected_sets = {
        "deep_sleep_states": {str(item["deep_sleep"]) for item in snapshots},
        "throttle_statuses": {
            str(item["throttle_status"]) for item in snapshots
        },
        "performance_levels": {
            str(item["performance_level"]) for item in snapshots
        },
    }
    for field_name, expected in expected_sets.items():
        observed = _audit_string_set(summary.get(field_name), field_name)
        if observed != expected:
            raise ValueError(
                "architecture audit telemetry summary state mismatch",
            )
    numeric = _audit_mapping(
        summary.get("numeric"),
        "telemetry numeric summary",
    )
    for field_name in _TELEMETRY_NUMERIC_FIELDS:
        statistics_payload = _audit_mapping(
            numeric.get(field_name),
            f"telemetry {field_name} summary",
        )
        values = [
            _positive_number(
                snapshot.get(field_name),
                f"telemetry {field_name}",
            )
            for snapshot in snapshots
        ]
        _close(
            statistics_payload.get("minimum"),
            min(values),
            "telemetry minimum",
        )
        _close(
            statistics_payload.get("median"),
            statistics.median(values),
            "telemetry median",
        )
        _close(
            statistics_payload.get("maximum"),
            max(values),
            "telemetry maximum",
        )


def _verify_statistics(
    payload: Mapping[str, object],
    *,
    samples: tuple[float, ...],
    batch_medians: tuple[float, ...],
    expected_bootstrap_replicates: int,
    expected_bootstrap_seed: int,
) -> None:
    if payload.get("primary_statistic") != "median_of_process_batch_medians":
        raise ValueError("architecture audit statistics primary mismatch")
    mean = statistics.mean(samples)
    standard_deviation = statistics.stdev(samples)
    lower_quartile = _quantile(samples, 0.25)
    upper_quartile = _quantile(samples, 0.75)
    comparisons = (
        ("primary_result", statistics.median(batch_medians)),
        ("all_sample_median", statistics.median(samples)),
        ("all_sample_mean", mean),
        ("all_sample_standard_deviation", standard_deviation),
        ("coefficient_of_variation", standard_deviation / mean),
        ("minimum", min(samples)),
        ("maximum", max(samples)),
        ("lower_quartile", lower_quartile),
        ("upper_quartile", upper_quartile),
        ("interquartile_range", upper_quartile - lower_quartile),
    )
    for field_name, expected in comparisons:
        _close(payload.get(field_name), expected, f"statistics {field_name}")
    _verify_batch_medians(payload.get("batch_medians"), batch_medians)
    _verify_confidence_interval(
        payload.get("bootstrap_median_confidence_interval_95"),
        primary=statistics.median(batch_medians),
        expected_replicates=expected_bootstrap_replicates,
        expected_seed=expected_bootstrap_seed,
    )


def _verify_batch_medians(
    value: object,
    expected_medians: tuple[float, ...],
) -> None:
    claimed = _number_sequence(value, "batch medians")
    if len(claimed) != len(expected_medians):
        raise ValueError("architecture audit batch median count mismatch")
    for claimed_value, expected in zip(claimed, expected_medians, strict=True):
        _close(claimed_value, expected, "batch median")


def _verify_confidence_interval(
    value: object,
    *,
    primary: float,
    expected_replicates: int,
    expected_seed: int,
) -> None:
    interval = _audit_mapping(value, "bootstrap confidence interval")
    lower = _positive_number(interval.get("lower"), "confidence lower bound")
    upper = _positive_number(interval.get("upper"), "confidence upper bound")
    if (
        lower > primary
        or upper < primary
        or interval.get("confidence_level") != 0.95
        or interval.get("method") != _BOOTSTRAP_METHOD
        or interval.get("replicates") != expected_replicates
        or interval.get("seed") != expected_seed
    ):
        raise ValueError("architecture audit confidence interval is invalid")


def _verify_measurement_tuning(
    measurement: Mapping[str, object],
    *,
    expected_tuning_batches: int,
) -> None:
    tuning = _audit_mapping(measurement.get("tuning"), "measurement tuning")
    selected = _audit_mapping(
        measurement.get("selected_configuration"),
        "selected configuration",
    )
    if tuning.get("status") == "not_required":
        if selected or tuning.get("selected_configuration") != {}:
            raise ValueError(
                "architecture audit untuned probe has a configuration",
            )
        return
    _verify_tuning_protocol(tuning)
    parameter = str(tuning.get("parameter", ""))
    raw_values = tuning.get("candidate_values")
    raw_candidates = tuning.get("candidates")
    if (
        not parameter
        or not isinstance(raw_values, list)
        or not isinstance(raw_candidates, list)
        or len(raw_values) != len(raw_candidates)
    ):
        raise ValueError("architecture audit tuning search space is invalid")
    candidate_values = tuple(
        _positive_int(value, "tuning candidate value") for value in raw_values
    )
    if len(set(candidate_values)) != len(candidate_values):
        raise ValueError("architecture audit tuning candidates must be unique")
    results = _verify_tuning_candidates(
        raw_candidates,
        parameter=parameter,
        expected_tuning_batches=expected_tuning_batches,
        expected_samples_per_batch=_positive_int(
            measurement.get("samples_per_process_batch"),
            "samples per process batch",
        ),
    )
    if tuple(result[1] for result in results) != candidate_values:
        raise ValueError("architecture audit tuning candidate order mismatch")
    _verify_tuning_execution_order(
        tuning.get("execution_order"),
        parameter=parameter,
        candidate_values=candidate_values,
        expected_tuning_batches=expected_tuning_batches,
    )
    selected_result = max(results, key=lambda result: (result[0], -result[1]))
    _verify_frozen_tuning(
        measurement,
        tuning,
        selected,
        parameter,
        selected_result,
    )


def _verify_tuning_protocol(tuning: Mapping[str, object]) -> None:
    if (
        tuning.get("status") != "performed"
        or tuning.get("phase") != "configuration_selection_only"
        or tuning.get("search_method") != "deterministic exhaustive search"
        or tuning.get("held_out_samples_used_for_selection") is not False
    ):
        raise ValueError("architecture audit tuning protocol is invalid")


def _verify_tuning_candidates(
    values: Sequence[object],
    *,
    parameter: str,
    expected_tuning_batches: int,
    expected_samples_per_batch: int,
) -> tuple[tuple[float, int, Mapping[str, object], str], ...]:
    return tuple(
        _verify_tuning_candidate(
            value,
            parameter=parameter,
            expected_tuning_batches=expected_tuning_batches,
            expected_samples_per_batch=expected_samples_per_batch,
        )
        for value in values
    )


def _verify_tuning_candidate(
    value: object,
    *,
    parameter: str,
    expected_tuning_batches: int,
    expected_samples_per_batch: int,
) -> tuple[float, int, Mapping[str, object], str]:
    candidate = _audit_mapping(value, "tuning candidate")
    configuration = _audit_mapping(
        candidate.get("configuration"),
        "tuning candidate configuration",
    )
    if set(configuration) != {parameter}:
        raise ValueError(
            "architecture audit tuning candidate configuration mismatch",
        )
    candidate_value = _positive_int(
        configuration.get(parameter),
        "tuning candidate value",
    )
    batches, samples, _ = _raw_batches(
        candidate.get("raw_process_batches"),
        expected_batches=expected_tuning_batches,
        expected_samples_per_batch=expected_samples_per_batch,
        require_telemetry=False,
    )
    batch_medians = tuple(statistics.median(batch) for batch in batches)
    selection_result = statistics.median(batch_medians)
    if candidate.get(
        "process_batch_count",
    ) != expected_tuning_batches or candidate.get(
        "sample_count",
    ) != len(samples):
        raise ValueError("architecture audit tuning sample count mismatch")
    _close(
        candidate.get("selection_result"),
        selection_result,
        "tuning selection result",
    )
    _close(
        candidate.get("all_sample_median"),
        statistics.median(samples),
        "tuning all-sample median",
    )
    _close(
        candidate.get("interquartile_range"),
        _quantile(samples, 0.75) - _quantile(samples, 0.25),
        "tuning interquartile range",
    )
    _verify_batch_medians(candidate.get("batch_medians"), batch_medians)
    compiler_defines = _verify_candidate_defines(
        candidate.get("compiler_defines"),
        candidate_value,
    )
    binary_sha256 = _audit_sha256(
        candidate.get("binary_sha256"),
        "tuning binary SHA-256",
    )
    return selection_result, candidate_value, compiler_defines, binary_sha256


def _verify_candidate_defines(
    value: object,
    candidate_value: int,
) -> Mapping[str, object]:
    compiler_defines = _audit_mapping(value, "tuning compiler defines")
    if (
        len(compiler_defines) != 1
        or not all(str(name).strip() for name in compiler_defines)
        or tuple(compiler_defines.values()) != (candidate_value,)
    ):
        raise ValueError("architecture audit tuning compiler defines mismatch")
    return compiler_defines


def _verify_tuning_execution_order(
    value: object,
    *,
    parameter: str,
    candidate_values: tuple[int, ...],
    expected_tuning_batches: int,
) -> None:
    if not isinstance(value, list):
        raise ValueError("architecture audit tuning execution order is missing")
    candidate_count = len(candidate_values)
    if len(value) != candidate_count * expected_tuning_batches:
        raise ValueError(
            "architecture audit tuning execution order is incomplete",
        )
    for tuning_round in range(expected_tuning_batches):
        start = tuning_round * candidate_count
        raw_round = value[start : start + candidate_count]
        expected_order = (
            candidate_values
            if tuning_round % 2 == 0
            else tuple(reversed(candidate_values))
        )
        for raw_entry, expected_value in zip(
            raw_round,
            expected_order,
            strict=True,
        ):
            entry = _audit_mapping(raw_entry, "tuning execution order entry")
            if (
                entry.get("tuning_round") != tuning_round
                or entry.get(parameter) != expected_value
            ):
                raise ValueError(
                    "architecture audit tuning execution order is invalid",
                )


def _verify_frozen_tuning(
    measurement: Mapping[str, object],
    tuning: Mapping[str, object],
    selected: Mapping[str, object],
    parameter: str,
    selected_result: tuple[float, int, Mapping[str, object], str],
) -> None:
    _, expected_value, expected_defines, expected_binary_sha = selected_result
    if selected != {parameter: expected_value}:
        raise ValueError(
            "architecture audit selected tuning candidate mismatch",
        )
    if tuning.get("selected_configuration") != dict(selected):
        raise ValueError(
            "architecture audit frozen tuning configuration mismatch",
        )
    compiler_defines = _audit_mapping(
        measurement.get("compiler_defines"),
        "compiler defines",
    )
    if compiler_defines != expected_defines:
        raise ValueError("architecture audit frozen compiler defines mismatch")
    if measurement.get("binary_sha256") != expected_binary_sha:
        raise ValueError("architecture audit frozen binary identity mismatch")


def _positive_int(value: object, field: str) -> int:
    numeric = _audit_nonnegative_int(value, field)
    if numeric == 0:
        raise ValueError(
            f"architecture audit evidence {field} must be a positive integer",
        )
    return numeric


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"architecture audit evidence {field} must be numeric")
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(
            f"architecture audit evidence {field} must be numeric",
        ) from exc
    if not math.isfinite(numeric):
        raise ValueError(f"architecture audit evidence {field} must be finite")
    return numeric


def _positive_number(value: object, field: str) -> float:
    numeric = _finite_number(value, field)
    if numeric <= 0:
        raise ValueError(
            f"architecture audit evidence {field} must be positive and finite",
        )
    return numeric


def _number_sequence(value: object, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"architecture audit evidence {field} must be a nonempty list",
        )
    return tuple(_positive_number(item, f"{field} item") for item in value)


def _quantile(values: tuple[float, ...], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _close(value: object, expected: float, field: str) -> None:
    actual = _finite_number(value, field)
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"architecture audit evidence {field} mismatch")
