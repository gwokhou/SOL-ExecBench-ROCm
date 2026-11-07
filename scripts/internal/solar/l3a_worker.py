"""One isolated L3-a measurement or pilot worker subprocess."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.internal.solar.l3a_measure import MapperMonitor, StageCollector

from solar.schema_versions import SchemaVersion

MESSAGE_LIMIT = 4096


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    measure = subparsers.add_parser("measure")
    measure.add_argument("--slot", required=True)
    measure.add_argument("--problem-dir", required=True, type=Path)
    measure.add_argument("--workload-uuid", required=True)
    measure.add_argument("--ir-path", required=True)
    measure.add_argument("--device", required=True)
    measure.add_argument("--orojenesis-home", required=True, type=Path)
    measure.add_argument("--record", required=True, type=Path)

    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--slot", required=True)
    pilot.add_argument("--problem-dir", required=True, type=Path)
    pilot.add_argument("--workload-uuid", required=True)
    pilot.add_argument("--device", required=True)
    pilot.add_argument("--record", required=True, type=Path)

    focused = subparsers.add_parser("focused")
    focused.add_argument("--device", default="cpu")
    focused.add_argument("--record-dir", required=True, type=Path)
    return parser


def run_measure(args: argparse.Namespace, torch_seconds: float) -> None:
    """Drive the real ``solar.api.analyze`` flow with stage probes."""
    from scripts.internal.solar.l3a_measure import (
        GpuPeakMonitor,
        MapperMonitor,
        MeasuredStage,
        StageCollector,
        install_stage_probes,
        stage_probe,
        timed_orojenesis_runner,
    )
    from sol_execbench.core.bench.gpu_execution import native_gpu_execution
    from sol_execbench.core.solar_bridge.analyzer import require_formal_device
    from sol_execbench.core.solar_bridge.formal_device import (
        FORMAL_ARCHITECTURE,
        device_execution_policy,
    )
    from sol_execbench.core.solar_bridge.models import (
        formal_precision_for_definition,
    )
    from sol_execbench.core.solar_bridge.semantic_metadata import (
        performance_analysis_metadata,
    )
    from sol_execbench.core.solar_bridge.workload_context import (
        load_solar_workload_context,
        solar_conversion_request,
    )
    from solar.api import AnalysisRequest, analyze
    from solar.ir.contracts import IRPath

    device = args.device
    ir_path = IRPath(args.ir_path)
    collector = StageCollector()
    gpu = GpuPeakMonitor(device)
    mapper = MapperMonitor()
    outcome: dict[str, Any] = {}
    with native_gpu_execution(device):
        if device.split(":", 1)[0] != "cpu":
            require_formal_device(device)
        with stage_probe(collector, MeasuredStage.WORKLOAD_CONTEXT_LOAD, gpu):
            context = load_solar_workload_context(
                args.problem_dir, args.workload_uuid, device
            )
    wrapped = replace(
        context,
        reference=collector.wrap_reference(context.reference),
        input_factory=collector.wrap_input_factory(context.input_factory),
    )
    request = AnalysisRequest(
        conversion=solar_conversion_request(wrapped, device, ir_path),
        architecture=FORMAL_ARCHITECTURE,
        output_dir=Path(args.record).parent / "analysis",
        precision=formal_precision_for_definition(context.definition),
        require_orojenesis=True,
        require_verified_audit=True,
        orojenesis_home=args.orojenesis_home,
        analysis_metadata=performance_analysis_metadata(context),
        execution_policy=device_execution_policy(device, None, 60.0),
    )
    from solar.analysis import graph_analyzer

    original_runner = graph_analyzer.OrojenesisRunner
    graph_analyzer.OrojenesisRunner = timed_orojenesis_runner(mapper)
    restore = install_stage_probes(collector, gpu)
    try:
        result = analyze(request)
    finally:
        restore()
        graph_analyzer.OrojenesisRunner = original_runner
        outcome = _outcome_dict(result)
        _attach_reason_code(collector.rows, result)
    with stage_probe(collector, MeasuredStage.INPUT_GENERATION_STANDALONE, gpu):
        request.input_factory(request.trace_seed)
    _write_measurement(
        args, torch_seconds, ir_path.value, collector, mapper, outcome
    )


def _write_measurement(
    args: argparse.Namespace,
    torch_seconds: float,
    ir_path: str,
    collector: StageCollector,
    mapper: MapperMonitor,
    outcome: dict[str, Any],
) -> None:
    """Publish the observed stages, mapper use, and outcome for one slot."""
    from scripts.internal.solar.l3a_measure import peak_rss_kb
    from sol_execbench.core.data.json_utils import atomic_write_json_value

    record = {
        "schema_version": SchemaVersion.SOLAR_L3A_MEASUREMENT,
        "slot": args.slot,
        "ir_path": ir_path,
        "device": args.device,
        "torch_import_seconds": torch_seconds,
        "process_peak_rss_kb": peak_rss_kb(),
        "stages": collector.rows,
        "attribution": collector.attribution,
        "mapper": {
            "runs": mapper.runs,
            "seconds": mapper.seconds,
            "peak_rss_kb": mapper.peak_rss_kb,
        },
        "outcome": outcome,
    }
    atomic_write_json_value(Path(args.record), record)


def _outcome_dict(result: Any) -> dict[str, Any]:
    return {
        "status": str(result.status),
        "stage": str(getattr(result, "stage", "") or "") or None,
        "reason_code": getattr(result, "reason_code", None),
        "message": str(getattr(result, "message", "") or "")[:MESSAGE_LIMIT],
        "publication_eligible": getattr(result, "publication_eligible", None),
        "bound_seconds": getattr(
            getattr(result, "bound", None), "seconds", None
        ),
    }


def _attach_reason_code(rows: dict[str, Any], result: Any) -> None:
    stage = str(getattr(result, "stage", "") or "")
    reason = getattr(result, "reason_code", None)
    row = rows.get(stage)
    if reason is not None and row is not None:
        row["reason_code"] = reason


def main() -> int:
    """Dispatch one bounded worker request; failures keep partial rows."""
    args = _argument_parser().parse_args()
    start = perf_counter()
    import torch  # noqa: F401 -- measured import side effect

    torch_seconds = perf_counter() - start
    if args.mode == "measure":
        run_measure(args, torch_seconds)
    elif args.mode == "pilot":
        from scripts.internal.solar.l3a_worker_pilot import run_pilot

        run_pilot(args, torch_seconds)
    else:
        from scripts.internal.solar.l3a_worker_pilot import run_focused

        run_focused(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
