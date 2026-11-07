"""Run full CPU analyses for the batch-1 flipped rows (L3-e gate G4).

Each flipped row gets one isolated subprocess running the production
``analyze`` on CPU with the pinned mapper. The gate requires >= 40/46
analyzed with formal totals; every miss keeps its production reason code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.timestamps import utc_timestamp
from solar.artifacts import sha256_file
from solar.schema_versions import SchemaVersion

REPO_ROOT = Path(__file__).resolve().parents[3]
G4_FLOOR = 40


def run_slot(args: argparse.Namespace) -> None:
    """Analyze one flipped slot on CPU and reduce it to a record."""
    import yaml

    from sol_execbench.core.solar_bridge.formal_device import (
        FORMAL_ARCHITECTURE,
        device_execution_policy,
    )
    from sol_execbench.core.solar_bridge.models import (
        formal_precision_for_definition,
    )
    from sol_execbench.core.solar_bridge.workload_context import (
        load_solar_workload_context,
        solar_conversion_request,
    )
    from solar.api import AnalysisRequest, analyze
    from solar.ir.contracts import IRPath

    item, ir_path = args.item.rsplit(":", 1)
    problem_dir = (
        REPO_ROOT / "problems" / "AMD_AKA" / "/".join(item.split("/")[:-1])
    )
    workload_uuid = item.split("/")[-1]
    context = load_solar_workload_context(problem_dir, workload_uuid, "cpu")
    request = AnalysisRequest(
        conversion=solar_conversion_request(context, "cpu", IRPath(ir_path)),
        architecture=FORMAL_ARCHITECTURE,
        output_dir=args.output / "analysis",
        precision=formal_precision_for_definition(context.definition),
        require_orojenesis=True,
        require_verified_audit=True,
        orojenesis_home=args.orojenesis_home,
        execution_policy=device_execution_policy("cpu", None, 60.0),
    )
    import shutil

    shutil.rmtree(args.output / "analysis", ignore_errors=True)
    result = analyze(request)
    record: dict[str, Any] = {
        "schema_version": SchemaVersion.SOLAR_L3E_ANALYSIS,
        "item": item,
        "ir_path": ir_path,
        "status": result.status.value,
        "reason_code": getattr(result, "reason_code", None),
        "message": str(getattr(result, "message", "") or "")[:2048],
    }
    analysis = args.output / "analysis" / "solar-analysis.yaml"
    if result.status.value == "analyzed" and analysis.is_file():
        document = yaml.safe_load(analysis.read_text())
        totals = document.get("total", {})
        record["totals"] = {
            key: totals.get(key)
            for key in ("macs", "flops", "other_ops", "num_layers")
        }
    atomic_write_json_value(args.output / "record.json", record)


def main() -> int:
    """Analyze all flipped slots and evaluate the pre-fixed G4 gate."""
    args = _parse_args()
    if args.item is not None:
        run_slot(args)
        return 0
    records = _run_campaign_slots(args)
    analyzed = [
        key
        for key, record in records.items()
        if record.get("status") == "analyzed" and record.get("totals")
    ]
    misses = [
        {
            "item": key,
            "status": record.get("status"),
            "reason": record.get("reason_code") or record.get("worker_error"),
        }
        for key, record in sorted(records.items())
        if key not in set(analyzed)
    ]
    revision = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    summary = {
        "schema_version": SchemaVersion.SOLAR_L3E_SUMMARY,
        "generated_at": utc_timestamp(),
        "identity": {
            "git_commit": revision,
            "mapper_sha256": sha256_file(
                args.orojenesis_home / "bin" / "timeloop-mapper"
            ),
            "flipped_sha256": sha256_file(args.flipped),
        },
        "g4": {
            "analyzed": len(analyzed),
            "total": len(records),
            "floor": args.floor or len(records),
            "passed": len(analyzed) >= (args.floor or len(records)),
            "misses": misses,
        },
    }
    atomic_write_json_value(args.output_root / "l3e-g4-summary.json", summary)
    print(json.dumps(summary["g4"], indent=1)[:800])
    return 0 if summary["g4"]["passed"] else 1


def _parse_args() -> argparse.Namespace:
    """Parse campaign inputs and optional single-slot execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flipped", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--orojenesis-home",
        type=Path,
        default=Path(
            "/home/guohao/PyCharmMiscProject/CrossSOL-Eval/out/"
            "paper-a-local-20261005-v8/infrastructure-amd-v8/"
            "prerequisites/external/amd-orojenesis"
        ),
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--floor",
        type=int,
        default=0,
        help="minimum analyzed rows; 0 requires all rows to analyze",
    )
    parser.add_argument("--item", help="single-slot worker mode item")
    parser.add_argument("--output", type=Path, help="single-slot output dir")
    return parser.parse_args()


def _run_campaign_slots(args: argparse.Namespace) -> dict[str, Any]:
    """Collect each flipped slot with the existing bounded worker protocol."""
    from sol_execbench.core.process.subprocesses import (
        run_in_process_group_to_files,
    )

    flipped = json.loads(args.flipped.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for index, key in enumerate(sorted(flipped)):
        item, ir_path = key.rsplit(":", 1)
        slot = args.output_root / ir_path / "/".join(item.split("/")[-2:])
        record_file = slot / "record.json"
        if args.resume and record_file.is_file():
            records[key] = json.loads(record_file.read_text())
            continue
        slot.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "scripts.internal.solar.l3e_batch1_campaign",
            "--item",
            key,
            "--output",
            str(slot),
            "--orojenesis-home",
            str(args.orojenesis_home),
        ]
        error = None
        try:
            run_in_process_group_to_files(
                command,
                slot / "worker.out.log",
                slot / "worker.err.log",
                cwd=REPO_ROOT,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            error = "worker timeout"
        if error is not None or not record_file.is_file():
            atomic_write_json_value(
                record_file,
                {"item": item, "ir_path": ir_path, "worker_error": error},
            )
        records[key] = json.loads(record_file.read_text())
        print(
            f"  {index + 1}/{len(flipped)} {key} -> "
            f"{records[key].get('status')}",
            flush=True,
        )
    return records


if __name__ == "__main__":
    raise SystemExit(main())
