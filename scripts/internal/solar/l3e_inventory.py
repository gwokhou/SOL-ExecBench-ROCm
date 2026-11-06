"""Inventory the exact unproved op classes behind the L3-c external-reads gap.

For every matrix row that failed formal analysis with unproved external
reads, re-extract and re-convert on CPU through the production stages, then
run the external-reads blocker per layer to collect (path, target, reason).
Diagnostic only: nothing here changes proof semantics.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from sol_execbench.core.solar_bridge.workload_context import (
    load_solar_workload_context,
    solar_conversion_request,
)
from solar.ir.access_artifact import load_access_document
from solar.ir.contracts import IRPath

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "problems" / "AMD_AKA" / "manifest.yaml"


def failing_items(matrix: Path, reason_prefix: str) -> list[tuple[str, str]]:
    """Read the L3-c matrix and return (item, ir_path) rows to inventory."""
    items = []
    for line in matrix.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["formal_analysis"].startswith(reason_prefix):
            items.append((row["item"], row["ir_path"]))
    return items


def blockers_for_call(
    reference: Any,
    inputs: tuple[Any, ...],
    ir_path: IRPath,
    staging: Path,
    name: str,
) -> list[tuple[str, str]]:
    """Collect per-layer external-read blockers for an arbitrary callable.

    Extraction plus conversion only (see _blockers for the memory rationale).
    """
    from solar.analysis.external_reads import (
        _blocking_reason,
        transparent_declared_output_aliases_external,
    )
    from solar.graph.extraction import extract_operator_graph
    from solar.ir.access import AccessReason
    from solar.ir.contracts import layer_operation
    from solar.ir.conversion import convert_operator_graph

    operator = extract_operator_graph(
        reference,
        inputs,
        device="cpu",
        output_dir=staging,
        name=name,
        extraction_kind=ir_path.extraction_kind,
    )
    ir_graph = convert_operator_graph(
        operator,
        output_dir=staging,
        ir_kind=ir_path.ir_kind,
    )
    import yaml

    graph = yaml.safe_load(ir_graph.path.read_text())
    access = load_access_document(ir_graph.path)
    layers = graph["layers"]
    producer_of: dict[str, str] = {}
    for layer_id, layer in layers.items():
        for output_name in (layer.get("tensor_names") or {}).get(
            "outputs"
        ) or []:
            producer_of[output_name] = layer_id
    declared = {
        name.rsplit(".", 1)[0]
        for name in graph.get("graph_signature", {}).get("user_outputs", [])
    }
    found = []
    for layer_id, layer in layers.items():
        if layer_id in access.transparent_layers:
            # Mirrors the batch-6 production gate: internal-materialized
            # declared-output aliases carry no external-read question.
            if transparent_declared_output_aliases_external(
                layer, declared, access
            ):
                found.append(("<transparent_alias>", "unproven_alias"))
            continue
        if layer_operation(layer).get("target") == "input":
            continue
        reason = _blocking_reason(layer, access, layers, producer_of)
        if reason is not None:
            target = str(layer_operation(layer).get("target", "?"))
            found.append((target, reason.value))
    for source_name in access.source_indices:
        if source_name not in access.tensors and source_name not in declared:
            found.append(
                (
                    "<input_reason>",
                    access.reasons.get(
                        source_name, AccessReason.UNKNOWN_READS
                    ).value,
                )
            )
    return found


def _blockers(item: str, ir_path: str) -> list[tuple[str, str]]:
    """Derive blockers from extraction plus conversion only.

    Verification replay is deliberately skipped: blocker attribution reads
    only the IR graph layers and the conversion-side access document, and
    CPU replay of explicit-unfold conv verification can exceed host memory.
    Every inventoried slot already passed verification in the L3-c matrix.
    """
    import torch

    torch.manual_seed(0)
    problem_dir = (
        REPO_ROOT / "problems" / "AMD_AKA" / "/".join(item.split("/")[:-1])
    )
    workload_uuid = item.split("/")[-1]
    selected = IRPath(ir_path)
    conversion = _conversion_request(problem_dir, workload_uuid, selected)
    with tempfile.TemporaryDirectory() as tmp:
        return blockers_for_call(
            conversion.reference,
            tuple(conversion.input_factory(conversion.trace_seed)),
            selected,
            Path(tmp),
            conversion.analysis_id,
        )


def _conversion_request(
    problem_dir: Path, workload_uuid: str, ir_path: IRPath
) -> Any:
    context = load_solar_workload_context(problem_dir, workload_uuid, "cpu")
    return solar_conversion_request(context, "cpu", ir_path)


def _run_one_subprocess(
    item: str, ir_path: str, logs: Path
) -> list[dict[str, Any]]:
    """Run one inventory slot in its own process (memory dies with it)."""
    import subprocess
    import sys

    from sol_execbench.core.process.subprocesses import (
        run_in_process_group_to_files,
    )

    logs.mkdir(parents=True, exist_ok=True)
    safe = (item + ":" + ir_path).replace("/", "__")
    record = logs / f"{safe}.json"
    if record.is_file():
        return json.loads(record.read_text())
    command = [
        sys.executable,
        "-m",
        "scripts.internal.solar.l3e_inventory",
        "--single",
        f"{item}:{ir_path}",
        "--record",
        str(record),
    ]
    try:
        run_in_process_group_to_files(
            command,
            logs / f"{safe}.out.log",
            logs / f"{safe}.err.log",
            cwd=REPO_ROOT,
            timeout=600.0,
        )
    except subprocess.TimeoutExpired:
        return [{"target": "WORKER_TIMEOUT", "reason": ""}]
    if not record.is_file():
        return [{"target": "WORKER_NO_RECORD", "reason": ""}]
    return json.loads(record.read_text())


def main() -> int:
    """Inventory all failing rows and print the frozen gap vocabulary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--single", help="item:ir_path worker mode")
    parser.add_argument("--record", type=Path, help="worker record output")
    args = parser.parse_args()
    if args.single:
        # RLIMIT_AS is deliberately NOT used: it caps virtual address space
        # (torch/ROCm map tens of GB VA at ~700 MB RSS) and breaks imports.
        # Per-slot process isolation plus the verification-free path is the
        # actual protection; a cgroup MemoryMax would guard RSS correctly.
        item, ir_path = args.single.rsplit(":", 1)
        blockers = _blockers(item, ir_path)
        args.record.write_text(
            json.dumps([{"target": t, "reason": r} for t, r in blockers])
        )
        return 0
    items = failing_items(args.matrix, "failed:formal_analysis_failed")
    print(f"inventorying {len(items)} rows in isolated per-slot processes ...")
    combined: Counter[tuple[str, str, str]] = Counter()
    per_item: dict[str, Any] = {}
    logs = args.output.parent / "inventory-logs"
    for index, (item, ir_path) in enumerate(items):
        blockers = _run_one_subprocess(item, ir_path, logs)
        per_item[f"{item}:{ir_path}"] = blockers
        for entry in blockers:
            combined[(ir_path, entry["target"], entry["reason"])] += 1
        if index % 16 == 0:
            print(f"  {index}/{len(items)}")
    summary = [
        {"ir_path": p, "target": t, "reason": r, "count": c}
        for (p, t, r), c in combined.most_common()
    ]
    args.output.write_text(
        json.dumps({"summary": summary, "items": per_item}, indent=1)
    )
    for entry in summary[:24]:
        print(entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
