# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Two-level Agent metrics and representativeness-drift measurements."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from statistics import fmean

from sol_execbench.core.dataset.corpus_models import (
    CorpusManifest,
    CorpusTargetViewManifest,
    GeneratedWorkloadRecord,
    GenerationDecisionStatus,
)
from sol_execbench.core.generalization.models import (
    FAST_THRESHOLDS,
    CellWorkloadResult,
    MetricEstimate,
    StratumMetrics,
    WorkloadDrift,
)

ResultMetric = Callable[[list[CellWorkloadResult]], float | None]


def summarize_results(
    results: Iterable[CellWorkloadResult],
    *,
    seed_digest: str,
    replicates: int,
) -> StratumMetrics:
    """Average workloads within Definition, then weight Definitions equally."""
    grouped = _group_results(results)
    return StratumMetrics(
        definition_count=len(grouped),
        workload_count=sum(len(rows) for rows in grouped.values()),
        compilation_rate=_estimate(
            grouped,
            _rate("compiled"),
            seed_digest=seed_digest,
            replicates=replicates,
        ),
        correctness_rate=_estimate(
            grouped,
            _rate("correct"),
            seed_digest=seed_digest,
            replicates=replicates,
        ),
        fast_p={
            str(value): _estimate(
                grouped,
                _fast_rate(value),
                seed_digest=seed_digest,
                replicates=replicates,
            )
            for value in FAST_THRESHOLDS
        },
        conditional_geomean_speedup=_estimate(
            grouped,
            _conditional_speedup,
            seed_digest=seed_digest,
            replicates=replicates,
        ),
        sol_score=_estimate(
            grouped,
            _conditional_sol,
            seed_digest=seed_digest,
            replicates=replicates,
        ),
        sol_coverage=_estimate(
            grouped,
            _sol_coverage,
            seed_digest=seed_digest,
            replicates=replicates,
        ),
    )


def paired_metric_delta(
    control: Iterable[CellWorkloadResult],
    target: Iterable[CellWorkloadResult],
    metric: ResultMetric,
    *,
    seed_digest: str,
    replicates: int,
) -> MetricEstimate:
    """Return paired target-minus-control delta over common Definitions."""
    left = _group_results(control)
    right = _group_results(target)
    values = {
        key: _difference(metric(right[key]), metric(left[key]))
        for key in sorted(left.keys() & right.keys())
    }
    return _estimate_values(values, seed_digest, replicates)


def workload_drift(
    manifest: CorpusManifest,
    source: CorpusTargetViewManifest,
    target: CorpusTargetViewManifest,
    *,
    source_target_id: str,
    target_target_id: str,
) -> WorkloadDrift:
    """Measure workload changes without interpreting them as distribution proof."""
    families = {
        entry.semantic_id: entry.operation_family.value
        for entry in manifest.entries
    }
    source_rows = _records_by_key(source)
    target_rows = _records_by_key(target)
    source_ids = {item.semantic_id for item in source.workloads}
    target_ids = {item.semantic_id for item in target.workloads}
    common_ids = source_ids & target_ids
    common_keys = sorted(source_rows.keys() & target_rows.keys())
    return WorkloadDrift(
        source_target_id=source_target_id,
        target_target_id=target_target_id,
        source_definition_count=len(source_ids),
        target_definition_count=len(target_ids),
        common_definition_count=len(common_ids),
        support_jaccard=_jaccard(source_ids, target_ids),
        skip_reason_counts=_skip_reasons(target),
        latent_slot_signature_equal=_slot_signatures_equal(
            source,
            target,
            common_ids,
        ),
        categorical_jsd_bits=_categorical_drift(
            source,
            target,
            families,
        ),
        axis_log2_shifts=_axis_shifts(
            source_rows,
            target_rows,
            common_keys,
        ),
        common_scale_ratios=_scale_ratios(
            source_rows,
            target_rows,
            common_ids,
        ),
        resource_fraction_shifts=_resource_shifts(
            source,
            target,
            source_rows,
            target_rows,
            common_keys,
        ),
    )


def correctness_metric(rows: list[CellWorkloadResult]) -> float:
    """Definition-level correctness rate."""
    return fmean(float(row.correct) for row in rows)


def fast_metric(threshold: float) -> ResultMetric:
    """Build a Definition-level fast-p metric."""
    return _fast_rate(threshold)


def speedup_metric(rows: list[CellWorkloadResult]) -> float | None:
    """Definition-level correct-only geometric mean speedup."""
    return _conditional_speedup(rows)


def _group_results(
    results: Iterable[CellWorkloadResult],
) -> dict[str, list[CellWorkloadResult]]:
    grouped: dict[str, list[CellWorkloadResult]] = defaultdict(list)
    for result in results:
        grouped[result.semantic_id].append(result)
    return dict(grouped)


def _rate(field: str) -> ResultMetric:
    return lambda rows: fmean(float(getattr(row, field)) for row in rows)


def _fast_rate(threshold: float) -> ResultMetric:
    return lambda rows: fmean(
        float(
            row.correct and row.speedup is not None and row.speedup >= threshold
        )
        for row in rows
    )


def _conditional_speedup(rows: list[CellWorkloadResult]) -> float | None:
    values = [row.speedup for row in rows if row.correct and row.speedup]
    if not values:
        return None
    return math.exp(fmean(math.log(value) for value in values))


def _conditional_sol(rows: list[CellWorkloadResult]) -> float | None:
    values = [
        row.sol_score
        for row in rows
        if row.correct and row.sol_score is not None
    ]
    return fmean(values) if values else None


def _sol_coverage(rows: list[CellWorkloadResult]) -> float:
    return fmean(
        float(row.correct and row.sol_score is not None) for row in rows
    )


def _estimate(
    grouped: dict[str, list[CellWorkloadResult]],
    metric: ResultMetric,
    *,
    seed_digest: str,
    replicates: int,
) -> MetricEstimate:
    values = {key: metric(rows) for key, rows in grouped.items()}
    return _estimate_values(values, seed_digest, replicates)


def _estimate_values(
    values: dict[str, float | None],
    seed_digest: str,
    replicates: int,
) -> MetricEstimate:
    observed = [value for value in values.values() if value is not None]
    if not observed:
        return MetricEstimate(value=None, ci_low=None, ci_high=None)
    estimate = fmean(observed)
    if replicates == 0:
        return MetricEstimate(value=estimate, ci_low=None, ci_high=None)
    samples = _bootstrap(values, seed_digest, replicates)
    return MetricEstimate(
        value=estimate,
        ci_low=_percentile(samples, 0.025),
        ci_high=_percentile(samples, 0.975),
    )


def _bootstrap(
    values: dict[str, float | None],
    seed_digest: str,
    replicates: int,
) -> list[float]:
    keys = sorted(values)
    rng = random.Random(int(seed_digest[:16], 16))
    samples = []
    for _ in range(replicates):
        draw = [values[rng.choice(keys)] for _ in keys]
        present = [value for value in draw if value is not None]
        if present:
            samples.append(fmean(present))
    return sorted(samples)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    index = min(len(values) - 1, int(fraction * len(values)))
    return values[index]


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _records_by_key(
    view: CorpusTargetViewManifest,
) -> dict[tuple[str, str], GeneratedWorkloadRecord]:
    return {(item.semantic_id, item.slot_id): item for item in view.workloads}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _skip_reasons(view: CorpusTargetViewManifest) -> dict[str, int]:
    counts = Counter(
        item.status.value
        for item in view.decisions
        if item.status is not GenerationDecisionStatus.GENERATED
    )
    return dict(sorted(counts.items()))


def _slot_signatures_equal(
    source: CorpusTargetViewManifest,
    target: CorpusTargetViewManifest,
    common_ids: set[str],
) -> bool:
    return _slot_signatures(source, common_ids) == _slot_signatures(
        target,
        common_ids,
    )


def _slot_signatures(
    view: CorpusTargetViewManifest,
    semantic_ids: set[str],
) -> set[tuple[str, ...]]:
    return {
        (
            row.semantic_id,
            row.slot_id,
            row.role.value,
            row.regime.value,
            row.serving_phase.value,
            row.binding.value,
            str(row.scale_numerator),
            str(row.scale_denominator),
        )
        for row in view.workloads
        if row.semantic_id in semantic_ids
    }


def _categorical_drift(
    source: CorpusTargetViewManifest,
    target: CorpusTargetViewManifest,
    families: dict[str, str],
) -> dict[str, float]:
    selectors = {
        "operation_family": lambda row: families[row.semantic_id],
        "regime": lambda row: row.regime.value,
        "phase": lambda row: row.serving_phase.value,
        "role": lambda row: row.role.value,
    }
    return {
        name: _jsd(
            Counter(selector(row) for row in source.workloads),
            Counter(selector(row) for row in target.workloads),
        )
        for name, selector in selectors.items()
    }


def _jsd(left: Counter[str], right: Counter[str]) -> float:
    keys = left.keys() | right.keys()
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total and not right_total:
        return 0.0
    result = 0.0
    for key in keys:
        p = left[key] / left_total if left_total else 0.0
        q = right[key] / right_total if right_total else 0.0
        middle = (p + q) / 2
        if p:
            result += p * math.log2(p / middle) / 2
        if q:
            result += q * math.log2(q / middle) / 2
    return result


def _axis_shifts(
    source: dict[tuple[str, str], GeneratedWorkloadRecord],
    target: dict[tuple[str, str], GeneratedWorkloadRecord],
    keys: list[tuple[str, str]],
) -> dict[str, tuple[float, ...]]:
    shifts: dict[str, list[float]] = defaultdict(list)
    for key in keys:
        left, right = source[key], target[key]
        for axis in sorted(left.axes.keys() & right.axes.keys()):
            shifts[axis].append(math.log2(right.axes[axis] / left.axes[axis]))
    return {axis: tuple(values) for axis, values in sorted(shifts.items())}


def _scale_ratios(
    source: dict[tuple[str, str], GeneratedWorkloadRecord],
    target: dict[tuple[str, str], GeneratedWorkloadRecord],
    semantic_ids: set[str],
) -> tuple[float, ...]:
    source_scales = {
        row.semantic_id: row.common_scale for row in source.values()
    }
    target_scales = {
        row.semantic_id: row.common_scale for row in target.values()
    }
    return tuple(
        target_scales[key] / source_scales[key] for key in sorted(semantic_ids)
    )


def _resource_shifts(
    source_view: CorpusTargetViewManifest,
    target_view: CorpusTargetViewManifest,
    source: dict[tuple[str, str], GeneratedWorkloadRecord],
    target: dict[tuple[str, str], GeneratedWorkloadRecord],
    keys: list[tuple[str, str]],
) -> dict[str, tuple[float, ...]]:
    values: dict[str, list[float]] = defaultdict(list)
    for key in keys:
        left = _resource_fractions(source_view, source[key])
        right = _resource_fractions(target_view, target[key])
        for name in left:
            values[name].append(right[name] - left[name])
    return {name: tuple(rows) for name, rows in sorted(values.items())}


def _resource_fractions(
    view: CorpusTargetViewManifest,
    record: GeneratedWorkloadRecord,
) -> dict[str, float]:
    resources = record.requirements.resources
    return {
        "reference_peak_over_capacity": (
            resources.reference_peak_bytes / view.capacity_class_bytes
        ),
        "max_tensor_utilization": (
            resources.max_tensor_bytes / view.target.max_tensor_bytes
        ),
        "reference_ipc_utilization": (
            resources.reference_ipc_bytes
            / view.target.reference_ipc_limit_bytes
        ),
    }


__all__ = [
    "correctness_metric",
    "fast_metric",
    "paired_metric_delta",
    "speedup_metric",
    "summarize_results",
    "workload_drift",
]
