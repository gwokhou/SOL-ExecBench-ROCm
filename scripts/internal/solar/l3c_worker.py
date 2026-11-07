"""One isolated L3-c formal-analysis worker subprocess."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from solar.schema_versions import SchemaVersion

MESSAGE_LIMIT = 4096


def _operator_inventory(graph_path: Path) -> dict[str, Any]:
    """Extract the op multiset and effect flags from an operator graph."""
    import yaml

    document = yaml.safe_load(graph_path.read_text())
    ops: list[str] = []
    has_alias = False
    has_mutation = False
    for name in document.get("layers", {}):
        layer = document["layers"][name]
        semantic = layer.get("semantic_op", {})
        target = semantic.get("exact_target") or layer.get("type")
        if layer.get("type") in {"start", "identity"} or target in {
            "input",
            "identity",
            None,
        }:
            continue
        ops.append(str(target))
        effects = semantic.get("effects") or {}
        has_alias = has_alias or bool(effects.get("aliases"))
        has_mutation = has_mutation or bool(effects.get("mutates"))
    return {"ops": ops, "has_alias": has_alias, "has_mutation": has_mutation}


def _analysis_totals(analysis_path: Path) -> dict[str, Any] | None:
    """Read the accounting totals from solar-analysis.yaml."""
    import yaml

    document = yaml.safe_load(analysis_path.read_text())
    totals = document.get("total")
    if not isinstance(totals, dict):
        return None
    return {
        key: totals.get(key)
        for key in ("macs", "flops", "other_ops", "num_layers")
    }


def run_slot(args: argparse.Namespace) -> None:
    """Run one production formal analysis and reduce it to a record."""
    from sol_execbench.core.data.json_utils import atomic_write_json_value
    from sol_execbench.core.solar_bridge.analyzer import analyze_workload

    output = args.output
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)
    outcome = analyze_workload(
        problem_dir=args.problem_dir,
        workload_uuid=args.workload_uuid,
        output_dir=output / "analysis",
        device=args.device,
        orojenesis_home=args.orojenesis_home,
        ir_path=args.ir_path,
    )
    record: dict[str, Any] = {
        "schema_version": SchemaVersion.SOLAR_L3C_ANALYSIS,
        "item": args.item,
        "ir_path": args.ir_path,
        "status": outcome.status.value,
        "reason_code": outcome.reason_code,
        "message": (outcome.message or "")[:MESSAGE_LIMIT],
    }
    analysis_dir = output / "analysis"
    graph_path = analysis_dir / "operator_graph.yaml"
    totals_path = analysis_dir / "solar-analysis.yaml"
    if outcome.status.value == "analyzed":
        inventory = _operator_inventory(graph_path)
        record.update(inventory)
        record["totals"] = _analysis_totals(totals_path)
    atomic_write_json_value(output / "record.json", record)


def main() -> int:
    """Dispatch one bounded analysis worker request."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    slot = subparsers.add_parser("analyze-slot")
    slot.add_argument("--item", required=True)
    slot.add_argument("--problem-dir", required=True, type=Path)
    slot.add_argument("--workload-uuid", required=True)
    slot.add_argument("--ir-path", required=True)
    slot.add_argument("--device", required=True)
    slot.add_argument("--orojenesis-home", required=True, type=Path)
    slot.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_slot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
