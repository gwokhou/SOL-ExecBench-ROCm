"""Measurement primitives and pre-fixed oracle gates for the L3-a campaign."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from enum import StrEnum
from statistics import median
from typing import Any

EXPECTED_R08_TORCHVIEW = "R08:torchview_extended_einsum"
EXPECTED_R08_MAKE_FX = "R08:make_fx_aten"
MAPPER_SAMPLE_SECONDS = 0.05
MAPPER_COMM_PREFIX = "timeloop"


class MeasuredStage(StrEnum):
    """Closed per-stage row vocabulary for the L3-a measurement report."""

    WORKLOAD_CONTEXT_LOAD = "workload_context_load"
    ARCHITECTURE_LOAD = "architecture_load"
    GRAPH_EXTRACTION = "graph_extraction"
    IR_CONVERSION = "ir_conversion"
    CONVERSION_VERIFICATION = "conversion_verification"
    FORMAL_ANALYSIS = "formal_analysis"
    PUBLICATION_EXTRACT_BOUND = "publication_extract_bound"
    PUBLICATION_FINISH_ARTIFACTS = "publication_finish_artifacts"
    PUBLICATION_MANIFEST = "publication_manifest"
    PUBLICATION_VERIFY_TREE = "publication_verify_tree"
    INPUT_GENERATION_STANDALONE = "input_generation_standalone"


PIPELINE_STAGES = tuple(MeasuredStage)[:-1]
STAT_STAGES = tuple(MeasuredStage)
# Spec erratum 2026-11-05: R08/make_fx fails formal-bound extraction with
# "external reads are unproved" (fail-closed formal policy). L1 evidence
# predating this campaign never claimed R08 priceable (12 of 14 slots).
EXPECTED_FAILURES = {
    EXPECTED_R08_TORCHVIEW: (
        MeasuredStage.GRAPH_EXTRACTION.value,
        "graph_extraction_failed",
    ),
    EXPECTED_R08_MAKE_FX: (
        MeasuredStage.PUBLICATION_EXTRACT_BOUND.value,
        "formal_analysis_failed",
    ),
}


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the linearly interpolated percentile of an ordered sample."""
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def aggregate_stages(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate per-stage wall/memory statistics across completed slots."""
    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for stage in STAT_STAGES:
        walls = [
            row["wall_seconds"]
            for record in records.values()
            if (row := record["stages"].get(stage.value)) is not None
            and row["status"] == "passed"
        ]
        if not walls:
            continue
        rss_peaks = [
            row["peak_rss_kb"]
            for record in records.values()
            if (row := record["stages"].get(stage.value)) is not None
            and row["status"] == "passed"
        ]
        aggregated[stage.value] = {
            "wall_seconds": {
                "min": min(walls),
                "median": median(walls),
                "mean": sum(walls) / len(walls),
                "p95": percentile(walls, 0.95),
                "max": max(walls),
            },
            "peak_rss_kb": {"max": max(rss_peaks)},
        }
    return aggregated


def evaluate_measurement_gate(
    records: Mapping[str, Mapping[str, Any]],
    denominator: Sequence[str],
    *,
    cuda_mode: bool,
) -> dict[str, Any]:
    """Evaluate the pre-fixed measurement validity gate (spec section 2)."""
    failures: list[str] = []
    if set(records) != set(denominator):
        failures.append(
            f"denominator_mismatch:{sorted(set(records) ^ set(denominator))}"
        )
    for slot, record in sorted(records.items()):
        stages = record["stages"]
        expected = EXPECTED_FAILURES.get(slot)
        if expected is None:
            missing = [
                stage.value
                for stage in PIPELINE_STAGES
                if stages.get(stage.value, {}).get("status") != "passed"
            ]
            if missing:
                failures.append(f"{slot}:pipeline_stages_not_passed:{missing}")
                continue
            for stage in PIPELINE_STAGES:
                row = stages[stage.value]
                if row["wall_seconds"] <= 0 or row["peak_rss_kb"] <= 0:
                    failures.append(f"{slot}:{stage.value}:nonpositive_measure")
                if cuda_mode and row.get("gpu_peak_allocated_bytes") is None:
                    failures.append(f"{slot}:{stage.value}:gpu_peak_missing")
            continue
        expected_stage, expected_reason = expected
        failed_rows = [
            name
            for name, row in stages.items()
            if row.get("status") == "failed"
        ]
        reason = record.get("outcome", {}).get("reason_code")
        if failed_rows != [expected_stage] or reason != expected_reason:
            failures.append(
                f"{slot}:expected_failure_mismatch:{failed_rows}:{reason}"
            )
            continue
        prefix, *_ = [
            position
            for position, stage in enumerate(PIPELINE_STAGES)
            if stage.value == expected_stage
        ]
        for stage in PIPELINE_STAGES[:prefix]:
            if stages.get(stage.value, {}).get("status") != "passed":
                failures.append(f"{slot}:prefix_stage_not_passed:{stage.value}")
    return {
        "passed": not failures,
        "failures": failures,
        "checked_slots": len(records),
    }


def semantic_projection(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Project one operator graph onto the pre-fixed tier-1 oracle (O1)."""
    layers = {}
    for name, layer in graph["layers"].items():
        layers[name] = {
            "type": layer.get("type"),
            "semantic_op": layer.get("semantic_op"),
            "tensor_shapes": layer.get("tensor_shapes"),
            "tensor_dtypes": layer.get("tensor_dtypes"),
            "source_input_index": layer.get("source_input_index"),
        }
    signature = graph.get("graph_signature", {})
    return {
        "layers": layers,
        "outputs": graph.get("outputs"),
        "user_inputs": signature.get("user_inputs"),
        "user_outputs": signature.get("user_outputs"),
    }


def projection_diff(
    real: Mapping[str, Any], fake: Mapping[str, Any]
) -> list[str]:
    """Return sorted tier-1 projection difference paths (empty when equal)."""

    def differences(left: Any, right: Any, path: str) -> list[str]:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            out = []
            for key in sorted(set(left) | set(right)):
                if key not in left or key not in right:
                    out.append(f"{path}/{key}")
                else:
                    out.extend(
                        differences(left[key], right[key], f"{path}/{key}")
                    )
            return out
        if left != right:
            return [path]
        return []

    return sorted(differences(real, fake, ""))


def evaluate_pilot_gate(
    slots: Mapping[str, Mapping[str, Any]],
    focused: Mapping[str, Mapping[str, Any]],
    *,
    structure_check: bool,
) -> dict[str, Any]:
    """Evaluate the pre-fixed adoption decision D1-D4 (spec section 5)."""
    slot_ids = sorted(slots)
    d1_slots = [
        slot
        for slot in slot_ids
        if slots[slot].get("o1_equal") and slots[slot].get("o3_passed")
    ]
    d2_samples = [
        sample for sample in sorted(focused) if focused[sample].get("o4_passed")
    ]
    faster = slower_gpu = 0
    for slot in slot_ids:
        record = slots[slot]
        real_wall = record.get("real", {}).get("wall_seconds")
        fake_wall = record.get("fake", {}).get("wall_seconds")
        real_gpu = record.get("real", {}).get("gpu_peak_allocated_bytes")
        fake_gpu = record.get("fake", {}).get("gpu_peak_allocated_bytes")
        if real_wall and fake_wall and fake_wall <= 0.7 * real_wall:
            faster += 1
        if (
            real_gpu is not None
            and fake_gpu is not None
            and fake_gpu <= real_gpu
        ):
            slower_gpu += 1
    d1 = len(d1_slots) == 8
    d2 = len(d2_samples) == 3
    d3 = faster >= 6 and slower_gpu >= 6
    d4 = structure_check
    return {
        "d1": {"passed": d1, "slots": d1_slots},
        "d2": {"passed": d2, "samples": d2_samples},
        "d3": {
            "passed": d3,
            "wall_le_0.7x_slots": faster,
            "gpu_peak_not_higher_slots": slower_gpu,
        },
        "d4": {"passed": d4},
        "adopt": d1 and d2 and d3 and d4,
    }


def reset_peak_rss() -> bool:
    """Reset the process peak-RSS high-water mark; report kernel support."""
    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
    except OSError:
        return False
    return True


def peak_rss_kb() -> int:
    """Return the current VmHWM peak-RSS reading in KiB."""
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM"):
                return int(line.split()[1])
    raise RuntimeError("VmHWM is unavailable in /proc/self/status")


class GpuPeakMonitor:
    """Reset and read the accelerator caching-allocator peak (torch.cuda API)."""

    def __init__(self, device: str) -> None:
        """Enable allocator tracking only for accelerator-family devices."""
        self.enabled = device.split(":", 1)[0] in {"cuda", "hip"}

    def reset(self) -> None:
        """Reset allocator peak counters for the measured device."""
        if not self.enabled:
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak(self) -> dict[str, int | None]:
        """Return the allocated and reserved peak bytes since the reset."""
        if not self.enabled:
            return {
                "gpu_peak_allocated_bytes": None,
                "gpu_peak_reserved_bytes": None,
            }
        import torch

        if not torch.cuda.is_available():
            return {
                "gpu_peak_allocated_bytes": None,
                "gpu_peak_reserved_bytes": None,
            }
        return {
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }


class StageCollector:
    """Collect per-stage rows plus reference/factory/mapper attribution."""

    def __init__(self) -> None:
        """Start with no rows and no active stage bucket."""
        self.rows: dict[str, dict[str, Any]] = {}
        self.current: str | None = None
        self.attribution: dict[str, dict[str, Any]] = {}

    def bucket(self, name: str) -> dict[str, Any]:
        """Return one attribution bucket, creating it on first use."""
        return self.attribution.setdefault(
            name,
            {
                "reference_calls": 0,
                "reference_seconds": 0.0,
                "input_factory_calls": 0,
                "input_factory_seconds": 0.0,
                "mapper_runs": 0,
                "mapper_seconds": 0.0,
                "mapper_peak_rss_kb": None,
            },
        )

    def wrap_reference[**P, R](
        self, function: Callable[P, R]
    ) -> Callable[P, R]:
        """Count and time every reference invocation per stage bucket."""

        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            bucket = self.bucket(self.current or "outside_stages")
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                bucket["reference_calls"] += 1
                bucket["reference_seconds"] += time.perf_counter() - start

        return wrapped

    def wrap_input_factory[**P, R](
        self, function: Callable[P, R]
    ) -> Callable[P, R]:
        """Count and time every input-factory invocation per stage bucket."""

        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            bucket = self.bucket(self.current or "outside_stages")
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                bucket["input_factory_calls"] += 1
                bucket["input_factory_seconds"] += time.perf_counter() - start

        return wrapped


@contextmanager
def stage_probe(
    collector: StageCollector,
    stage: MeasuredStage,
    gpu: GpuPeakMonitor,
) -> Iterator[None]:
    """Measure one stage: wall time, reset peak RSS, and allocator peak."""
    reset_ok = reset_peak_rss()
    gpu.reset()
    previous, collector.current = collector.current, stage.value
    start = time.perf_counter()
    status, message = "passed", None
    try:
        yield
    except Exception as exc:
        status, message = "failed", str(exc)[:4096]
        raise
    finally:
        collector.current = previous
        row: dict[str, Any] = {
            "stage": stage.value,
            "status": status,
            "wall_seconds": time.perf_counter() - start,
            "peak_rss_kb": peak_rss_kb(),
            "peak_rss_reset": reset_ok,
            **gpu.peak(),
        }
        if message is not None:
            row["message"] = message
        collector.rows[stage.value] = row


class MapperMonitor:
    """Attribute Orojenesis mapper wall time and sampled peak RSS."""

    def __init__(self) -> None:
        """Start with no observed mapper invocations."""
        self.runs = 0
        self.seconds = 0.0
        self.peak_rss_kb: int | None = None

    @contextmanager
    def measure(self) -> Iterator[None]:
        """Time one mapper invocation while sampling /proc for its peak."""
        stop = threading.Event()
        thread = threading.Thread(
            target=self._sample, args=(stop,), daemon=True
        )
        start = time.perf_counter()
        thread.start()
        try:
            yield
        finally:
            self.seconds += time.perf_counter() - start
            self.runs += 1
            stop.set()
            thread.join(timeout=2.0)

    def _sample(self, stop: threading.Event) -> None:
        maximum = 0
        while not stop.is_set():
            maximum = max(maximum, self._current_peak())
            stop.wait(MAPPER_SAMPLE_SECONDS)
        if maximum:
            self.peak_rss_kb = max(self.peak_rss_kb or 0, maximum)

    @staticmethod
    def _current_peak() -> int:
        best = 0
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                with open(f"/proc/{entry.name}/comm") as handle:
                    if not handle.read().startswith(MAPPER_COMM_PREFIX):
                        continue
                with open(f"/proc/{entry.name}/status") as handle:
                    for line in handle:
                        if line.startswith("VmHWM"):
                            best = max(best, int(line.split()[1]))
                            break
            except OSError:
                continue
        return best


def timed_orojenesis_runner(monitor: MapperMonitor) -> type:
    """Build an OrojenesisRunner subclass that feeds the monitor."""
    from pathlib import Path
    from subprocess import CompletedProcess

    from solar.analysis.orojenesis.runner import OrojenesisRunner

    class _TimedOrojenesisRunner(OrojenesisRunner):
        def _run_mapper(
            self,
            command: Sequence[str],
            *,
            cwd: Path,
            env: Mapping[str, str] | None = None,
        ) -> CompletedProcess[str]:
            with monitor.measure():
                return super()._run_mapper(command, cwd=cwd, env=env)

    return _TimedOrojenesisRunner


def _stage_probe_targets() -> tuple[tuple[Any, str, MeasuredStage], ...]:
    """Identify the production stage entry points measured by this campaign."""
    import solar.api as solar_api
    import solar.pipeline.analysis as pipeline_analysis
    from solar.rocm import architecture

    return (
        (
            pipeline_analysis,
            "extract_request_graph",
            MeasuredStage.GRAPH_EXTRACTION,
        ),
        (
            pipeline_analysis,
            "convert_request_graph",
            MeasuredStage.IR_CONVERSION,
        ),
        (
            pipeline_analysis,
            "verify_request_graph",
            MeasuredStage.CONVERSION_VERIFICATION,
        ),
        (
            pipeline_analysis,
            "analyze_request_graph",
            MeasuredStage.FORMAL_ANALYSIS,
        ),
        (
            solar_api,
            "extract_bound",
            MeasuredStage.PUBLICATION_EXTRACT_BOUND,
        ),
        (
            solar_api,
            "finish_artifacts",
            MeasuredStage.PUBLICATION_FINISH_ARTIFACTS,
        ),
        (
            solar_api,
            "write_request_manifest",
            MeasuredStage.PUBLICATION_MANIFEST,
        ),
        (
            solar_api,
            "verify_publication_tree",
            MeasuredStage.PUBLICATION_VERIFY_TREE,
        ),
        (
            architecture.ArchitectureProfile,
            "load",
            MeasuredStage.ARCHITECTURE_LOAD,
        ),
    )


def install_stage_probes(
    collector: StageCollector, gpu: GpuPeakMonitor
) -> Callable[[], None]:
    """Wrap production stage entry points; return a restore callable."""
    from solar.graph import make_fx_backend

    replacements: list[tuple[Any, str, Any]] = []

    def wrap[**P, R](module: Any, name: str, stage: MeasuredStage) -> None:
        original = getattr(module, name)
        replacements.append((module, name, original))

        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            with stage_probe(collector, stage, gpu):
                return original(*args, **kwargs)

        setattr(module, name, wrapped)

    for module, name, stage in _stage_probe_targets():
        wrap(module, name, stage)

    component = make_fx_backend.trace_make_fx_reference
    replacements.append((make_fx_backend, "trace_make_fx_reference", component))
    component_seconds = [0.0]

    def wrapped_trace(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return component(*args, **kwargs)
        finally:
            component_seconds[0] += time.perf_counter() - start
            collector.rows["component/make_fx_trace"] = {
                "stage": "component/make_fx_trace",
                "status": "passed",
                "wall_seconds_accumulated": component_seconds[0],
            }

    make_fx_backend.trace_make_fx_reference = wrapped_trace

    def restore() -> None:
        for module, name, original in replacements:
            setattr(module, name, original)

    return restore


__all__ = [
    "EXPECTED_R08_TORCHVIEW",
    "MAPPER_COMM_PREFIX",
    "MAPPER_SAMPLE_SECONDS",
    "PIPELINE_STAGES",
    "GpuPeakMonitor",
    "MapperMonitor",
    "StageCollector",
    "aggregate_stages",
    "evaluate_measurement_gate",
    "evaluate_pilot_gate",
    "install_stage_probes",
    "peak_rss_kb",
    "percentile",
    "projection_diff",
    "reset_peak_rss",
    "semantic_projection",
    "stage_probe",
    "timed_orojenesis_runner",
]
