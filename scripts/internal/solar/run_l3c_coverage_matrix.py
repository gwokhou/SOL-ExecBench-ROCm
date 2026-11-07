"""Run the L3-c corpus coverage-matrix campaign over both fixed paths.

Phase A audits every scored workload through the production readiness seam;
phase B runs formal analysis on ready slots; phase C assembles the frozen
coverage matrix, canary-wall attribution, dual-path differential, and gate.
Acceptance is pre-fixed in
docs/internal/solar-l3c-coverage-matrix-acceptance-2026-11-06.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.internal.solar.l3c_families import (
    ROADMAP_FAMILIES,
    attribute_failure,
    evaluate_l3c_gate,
    matrix_row,
    reconcile_canary,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.dataset.aka_corpus import (
    AKACorpusManifest,
    AKACorpusRole,
)
from sol_execbench.core.solar_bridge.corpus_readiness import (
    audit_corpus_stage_readiness,
)
from sol_execbench.core.timestamps import utc_timestamp
from solar.artifacts import sha256_file
from solar.ir.contracts import IRPath
from solar.schema_versions import SchemaVersion

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_ROOT = REPO_ROOT / "problems" / "AMD_AKA"
MANIFEST = CORPUS_ROOT / "manifest.yaml"
CANARY_MATRIX = (
    REPO_ROOT
    / "data/outputs/solar-l1-release-20261105/qualification/canary/"
    / "readiness/matrix.jsonl"
)
PATHS = (
    IRPath.MAKE_FX_ATEN,
    IRPath.TORCHVIEW_EXTENDED_EINSUM,
)
L1_CANARY_IR_PATH = "torchview_extended_einsum"


def _read_matrix(path: Path) -> dict[str, dict[str, Any]]:
    return {
        f"{record['problem_path']}/{record['workload_uuid']}": record
        for record in (
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        )
    }


def run_audit_phase(
    output: Path, args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    """Audit the full scored corpus per path via the production seam."""
    audits: dict[str, dict[str, Any]] = {}
    for ir_path in PATHS:
        root = output / "audit" / ir_path.value
        audit_corpus_stage_readiness(
            MANIFEST,
            root,
            device=args.device,
            timeout_seconds=3600.0,
            resume=args.resume,
            ir_path=ir_path,
        )
        audits[ir_path.value] = _read_matrix(root / "matrix.jsonl")
    return audits


def _scored_items() -> dict[str, tuple[Path, str]]:
    corpus = AKACorpusManifest.load(MANIFEST)
    items: dict[str, tuple[Path, str]] = {}
    for entry in corpus.entries:
        if entry.role is not AKACorpusRole.SCORED:
            continue
        problem = entry.relative_problem_dir.as_posix()
        for workload_uuid in entry.workload_uuids:
            items[f"{problem}/{workload_uuid}"] = (
                CORPUS_ROOT / problem,
                workload_uuid,
            )
    return items


def run_analysis_phase(
    output: Path, args: argparse.Namespace, audits: dict[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Run formal analysis on every ready slot through isolated workers."""
    records: dict[tuple[str, str], dict[str, Any]] = {}
    items = _scored_items()
    for ir_path in PATHS:
        for item_id in sorted(items):
            audit = audits[ir_path.value].get(item_id)
            if audit is None or audit.get("status") != "ready":
                continue
            problem_dir, workload_uuid = items[item_id]
            slot_output = (output / "analysis" / ir_path.value) / "/".join(
                item_id.split("/")[-2:]
            )
            record_file = slot_output / "record.json"
            if args.resume and record_file.is_file():
                records[(ir_path.value, item_id)] = json.loads(
                    record_file.read_text()
                )
                continue
            command = [
                sys.executable,
                "-m",
                "scripts.internal.solar.l3c_worker",
                "analyze-slot",
                "--item",
                item_id,
                "--problem-dir",
                str(problem_dir),
                "--workload-uuid",
                workload_uuid,
                "--ir-path",
                ir_path.value,
                "--device",
                args.device,
                "--orojenesis-home",
                str(args.orojenesis_home),
                "--output",
                str(slot_output),
            ]
            error = None
            try:
                _spawn(
                    command,
                    output / "worker-logs",
                    f"{ir_path.value}:{item_id}",
                    args.timeout,
                )
            except subprocess.TimeoutExpired:
                error = "worker timeout"
            if error is None and not record_file.is_file():
                error = "worker produced no record"
            if error is not None:
                atomic_write_json_value(
                    record_file,
                    {"item": item_id, "worker_error": error},
                )
            records[(ir_path.value, item_id)] = json.loads(
                record_file.read_text()
            )
    return records


def _spawn(command: list[str], logs: Path, slot: str, timeout: float) -> None:
    """Run one worker process group with bounded log files."""
    from sol_execbench.core.process.subprocesses import (
        run_in_process_group_to_files,
    )

    logs.mkdir(parents=True, exist_ok=True)
    safe = slot.replace("/", "__")
    run_in_process_group_to_files(
        command,
        logs / f"{safe}.out.log",
        logs / f"{safe}.err.log",
        cwd=REPO_ROOT,
        timeout=timeout,
    )


def assemble(
    output: Path,
    audits: dict[str, dict[str, Any]],
    analyses: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the matrix, attribution, differential, and gate verdict."""
    items = _scored_items()
    rows = []
    for ir_path in PATHS:
        for item_id in sorted(items):
            audit = audits[ir_path.value].get(item_id) or {
                "status": "worker_error"
            }
            analysis = analyses.get((ir_path.value, item_id))
            rows.append(matrix_row(item_id, ir_path.value, audit, analysis))
    with (output / "coverage-matrix.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    failed_rows = [
        row
        for row in rows
        if row["audit_status"] != "ready"
        or str(row["formal_analysis"]).startswith("failed")
    ]
    attribution = [attribute_failure(row) for row in failed_rows]

    canary = _read_matrix(CANARY_MATRIX)
    fresh = {
        item: audits[L1_CANARY_IR_PATH][item]
        for item in canary
        if item in audits.get(L1_CANARY_IR_PATH, {})
    }
    reconciliation = reconcile_canary(canary, fresh)

    differential = _path_differential(rows)

    family_summary: dict[str, Any] = {}
    for family in [*sorted(ROADMAP_FAMILIES), "elementwise", "other"]:
        tagged = [row for row in rows if family in row["families"]]
        per_path = Counter(row["ir_path"] for row in tagged)
        analyzed = sum(
            1 for row in tagged if row["formal_analysis"] == "analyzed"
        )
        family_summary[family] = {
            "rows": len(tagged),
            "analyzed": analyzed,
            "per_path": dict(per_path),
        }

    gate = evaluate_l3c_gate(rows, reconciliation, attribution)
    return {
        "rows": rows,
        "attribution": attribution,
        "reconciliation": reconciliation,
        "differential": differential,
        "family_summary": family_summary,
        "gate": gate,
    }


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "git_commit": revision,
        "corpus_manifest_sha256": sha256_file(MANIFEST),
        "mapper_sha256": sha256_file(
            args.orojenesis_home / "bin" / "timeloop-mapper"
        ),
        "device": args.device,
    }


def main() -> int:
    """Run the audit, analysis, and assembly phases; write the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--orojenesis-home",
        type=Path,
        default=Path(
            "/home/guohao/PyCharmMiscProject/CrossSOL-Eval/out/"
            "paper-a-local-20261005-v8/infrastructure-amd-v8/"
            "prerequisites/external/amd-orojenesis"
        ),
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    audits = run_audit_phase(args.output_root, args)
    analyses = run_analysis_phase(args.output_root, args, audits)
    assembled = assemble(args.output_root, audits, analyses)
    atomic_write_json_value(
        args.output_root / "canary-wall-attribution.json",
        {"failures": assembled["attribution"]},
    )
    atomic_write_json_value(
        args.output_root / "differential.json",
        {
            "both_analyzed": len(assembled["differential"]),
            "totals_equal": sum(
                1 for item in assembled["differential"] if item["totals_equal"]
            ),
            "items": assembled["differential"],
        },
    )
    atomic_write_json_value(
        args.output_root / "family-summary.json",
        assembled["family_summary"],
    )
    summary = {
        "schema_version": SchemaVersion.SOLAR_L3C_SUMMARY,
        "generated_at": utc_timestamp(),
        "identity": _identity(args),
        "gate": assembled["gate"],
        "reconciliation": assembled["reconciliation"],
        "row_count": len(assembled["rows"]),
        "failed_rows": len(assembled["attribution"]),
        "differential": {
            "both_analyzed": len(assembled["differential"]),
            "totals_equal": sum(
                1 for item in assembled["differential"] if item["totals_equal"]
            ),
        },
    }
    atomic_write_json_value(args.output_root / "l3c-summary.json", summary)
    print(
        json.dumps(
            {
                "passed": assembled["gate"]["passed"],
                "rows": summary["row_count"],
                "reconciliation": assembled["reconciliation"]["passed"],
            }
        )
    )
    return 0 if assembled["gate"]["passed"] else 1


def _path_differential(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare totals only for rows analyzed through both IR paths."""
    differential = []
    by_item: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_item.setdefault(row["item"], {})[row["ir_path"]] = row
    for item_id, per_path in sorted(by_item.items()):
        if set(per_path) != {path.value for path in PATHS}:
            continue
        left, right = (
            per_path[PATHS[0].value],
            per_path[PATHS[1].value],
        )
        if left["formal_analysis"] != "analyzed" or (
            right["formal_analysis"] != "analyzed"
        ):
            continue
        differential.append(
            {
                "item": item_id,
                "make_fx_totals": left["resource_totals"],
                "torchview_totals": right["resource_totals"],
                "totals_equal": (
                    left["resource_totals"] == right["resource_totals"]
                ),
            }
        )

    return differential


if __name__ == "__main__":
    raise SystemExit(main())
