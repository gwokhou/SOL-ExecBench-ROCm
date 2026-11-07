"""Fake-static pilot and focused-sample worker entry points for L3-a."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml

from scripts.internal.solar.l3a_measure import (
    GpuPeakMonitor,
    peak_rss_kb,
    projection_diff,
    reset_peak_rss,
    semantic_projection,
)
from scripts.internal.solar.l3a_samples import FOCUSED_SAMPLES
from solar.artifacts import sha256_file
from solar.graph.extraction import extract_operator_graph
from solar.ir.contracts import IRPath
from solar.schema_versions import SchemaVersion


def _trace_one(
    variant: str,
    reference: Any,
    inputs: tuple[Any, ...],
    output: Path,
    name: str,
    device: str,
) -> tuple[dict[str, Any], Any]:
    """Trace one variant (real or fake) with wall/RSS/GPU-peak measurement."""
    gpu = GpuPeakMonitor(device)
    reset_peak_rss()
    gpu.reset()
    start = time.perf_counter()
    if variant == "real":
        artifact = extract_operator_graph(
            reference,
            inputs,
            device=device,
            output_dir=output / "real",
            name=name,
            extraction_kind=IRPath.MAKE_FX_ATEN.extraction_kind,
        )
    else:
        # LineB: production adopted fake-static tracing; the historical
        # pilot's fake variant is the production backend itself.
        artifact = extract_operator_graph(
            reference,
            inputs,
            device=device,
            output_dir=output / "fake",
            name=name,
            extraction_kind=IRPath.MAKE_FX_ATEN.extraction_kind,
        )
    row = {
        "wall_seconds": time.perf_counter() - start,
        "peak_rss_kb": peak_rss_kb(),
        **gpu.peak(),
    }
    return row, artifact


def _load_document(artifact: Any) -> dict[str, Any]:
    return yaml.safe_load(Path(artifact.path).read_text())


def _verify_fake_graph(
    conversion_request: Any,
    fake_artifact: Any,
    output: Path,
) -> dict[str, Any]:
    """Run downstream conversion plus real replay on the fake graph.

    Conversion and verification share the fake artifact directory because
    the access-document loader requires the operator graph beside the IR
    graph, mirroring the production staging layout.
    """
    from solar.ir.conversion import convert_operator_graph
    from solar.pipeline.stages import verify_request_graph

    conversion_dir = Path(fake_artifact.path).parent
    try:
        graph = convert_operator_graph(
            fake_artifact,
            output_dir=conversion_dir,
            ir_kind=IRPath.MAKE_FX_ATEN.ir_kind,
        )
        verify_request_graph(
            conversion_request,
            graph,
            conversion_dir / "conversion-attestation.yaml",
        )
    except Exception as exc:  # noqa: BLE001 -- record the O3 failure
        return {"o3_passed": False, "o3_error": str(exc)[:4096]}
    return {
        "o3_passed": True,
        "operator_graph_sha256": sha256_file(fake_artifact.path),
        "aten_graph_sha256": sha256_file(graph.path),
    }


def run_pilot(args: argparse.Namespace, torch_seconds: float) -> None:
    """Compare real versus fake extraction for one make_fx corpus slot."""
    from sol_execbench.core.bench.gpu_execution import native_gpu_execution
    from sol_execbench.core.data.json_utils import atomic_write_json_value
    from sol_execbench.core.solar_bridge.analyzer import require_formal_device
    from sol_execbench.core.solar_bridge.workload_context import (
        load_solar_workload_context,
        solar_conversion_request,
    )

    device = args.device
    output = Path(args.record).parent
    with native_gpu_execution(device):
        if device.split(":", 1)[0] != "cpu":
            require_formal_device(device)
        context = load_solar_workload_context(
            args.problem_dir, args.workload_uuid, device
        )
    inputs = context.input_factory(200)
    record: dict[str, Any] = {
        "schema_version": SchemaVersion.SOLAR_L3A_PILOT,
        "slot": args.slot,
        "device": device,
        "torch_import_seconds": torch_seconds,
    }
    artifacts: dict[str, Any] = {}
    for variant in ("real", "fake"):
        try:
            row, artifact = _trace_one(
                variant,
                context.reference,
                inputs,
                output,
                args.workload_uuid,
                device,
            )
        except Exception as exc:  # noqa: BLE001 -- record the side failure
            record[f"{variant}_error"] = f"{type(exc).__name__}: {exc}"[:4096]
            record[variant] = {}
            continue
        record[variant] = row
        artifacts[variant] = artifact
    if {"real", "fake"} <= artifacts.keys():
        real_doc = _load_document(artifacts["real"])
        fake_doc = _load_document(artifacts["fake"])
        diff = projection_diff(
            semantic_projection(real_doc),
            semantic_projection(fake_doc),
        )
        record["o1_equal"] = not diff
        record["o1_diff"] = diff[:64]
        record["o2_exact_document_equal"] = real_doc == fake_doc
        conversion_request = solar_conversion_request(
            context, device, IRPath.MAKE_FX_ATEN
        )
        record.update(
            _verify_fake_graph(conversion_request, artifacts["fake"], output)
        )
    else:
        record["o1_equal"] = False
        record["o3_passed"] = False
    atomic_write_json_value(Path(args.record), record)


def run_focused(args: argparse.Namespace) -> None:
    """Evaluate the three focused samples against the pre-fixed O4 oracle."""
    from sol_execbench.core.data.json_utils import atomic_write_json_value

    device = args.device
    args.record_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("l3a_samples.py")
    source_sha256 = sha256_file(source)
    for sample in FOCUSED_SAMPLES:
        scratch = args.record_dir / sample.sample_id
        record = _focused_record(sample, device, source_sha256, scratch)
        atomic_write_json_value(
            args.record_dir / f"{sample.sample_id}.json", record
        )


def _focused_record(
    sample: Any,
    device: str,
    source_sha256: str,
    scratch: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": SchemaVersion.SOLAR_L3A_PILOT,
        "sample": sample.sample_id,
        "device": device,
    }
    artifacts: dict[str, Any] = {}
    for variant in ("real", "fake"):
        try:
            _, artifact = _trace_one(
                variant,
                sample.bound_reference(),
                sample.inputs(200),
                scratch,
                sample.sample_id,
                device,
            )
        except Exception as exc:  # noqa: BLE001 -- rejections are outcomes
            record[f"{variant}_error"] = f"{type(exc).__name__}: {exc}"[:4096]
            continue
        artifacts[variant] = artifact
    record["fake_rejected"] = "fake" not in artifacts
    if sample.sample_id == "data_dependent_branch":
        record["o4_passed"] = record["fake_rejected"]
        return record
    if not {"real", "fake"} <= artifacts.keys():
        record["o4_passed"] = False
        return record
    real_doc = _load_document(artifacts["real"])
    fake_doc = _load_document(artifacts["fake"])
    diff = projection_diff(
        semantic_projection(real_doc),
        semantic_projection(fake_doc),
    )
    record["o1_equal"] = not diff
    record["o1_diff"] = diff[:64]
    record["o2_exact_document_equal"] = real_doc == fake_doc
    conversion_request = sample.request(IRPath.MAKE_FX_ATEN, source_sha256)
    record.update(
        _verify_fake_graph(conversion_request, artifacts["fake"], scratch)
    )
    record["o4_passed"] = record["o1_equal"] and record["o3_passed"]
    return record


__all__ = ["run_focused", "run_pilot"]
