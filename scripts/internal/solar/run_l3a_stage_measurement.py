"""Run the L3-a stage-measurement campaign and fake-static pilot.

The measurement phase drives the frozen R16 denominator (both IR paths)
through one bounded worker subprocess per slot, mirroring the production
adapter chain. The pilot phase compares real versus fake-static extraction
on the eight make_fx slots plus the three focused samples, then evaluates
the pre-fixed D1-D4 decision rule. Acceptance is pre-fixed in
docs/internal/solar-l3a-fake-static-pilot-acceptance-2026-11-05.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.internal.solar.l3a_measure import (
    MEASUREMENT_SCHEMA,
    aggregate_stages,
    evaluate_measurement_gate,
    evaluate_pilot_gate,
)
from scripts.internal.solar.run_r16_curated_qualification import R16_ITEMS
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.process.subprocesses import (
    run_in_process_group_to_files,
)
from sol_execbench.core.timestamps import utc_timestamp
from solar.artifacts import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_ROOT = REPO_ROOT / "problems" / "AMD_AKA"
IR_PATH_VALUES = (
    "make_fx_aten",
    "torchview_extended_einsum",
)
PILOT_SCHEMA_SUMMARY = "sol_execbench.solar_l3a_pilot_summary.v1"


def _slot_ids() -> dict[str, tuple[Path, str, str]]:
    """Map each slot id to (problem_dir, workload_uuid, ir_path)."""
    slots: dict[str, tuple[Path, str, str]] = {}
    for item, path in R16_ITEMS.items():
        problem, workload = path.rsplit("/", 1)
        for ir_path in IR_PATH_VALUES:
            slots[f"{item}:{ir_path}"] = (
                CORPUS_ROOT / problem,
                workload,
                ir_path,
            )
    return slots


def _spawn(command: list[str], logs: Path, slot: str, timeout: float) -> None:
    """Run one worker process group with bounded log files."""
    logs.mkdir(parents=True, exist_ok=True)
    run_in_process_group_to_files(
        command,
        logs / f"{slot}.out.log",
        logs / f"{slot}.err.log",
        cwd=REPO_ROOT,
        timeout=timeout,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _resume_record(path: Path) -> Any | None:
    """Return a completed worker record for resume, else None."""
    if not path.is_file():
        return None
    record = _load_json(path)
    return record if "worker_error" not in record else None


def run_measure_phase(
    output: Path, args: argparse.Namespace, identity: dict[str, Any]
) -> dict[str, Any]:
    """Run all sixteen measurement slots and evaluate the validity gate."""
    slots = _slot_ids()
    records: dict[str, Any] = {}
    for slot, (problem_dir, workload, ir_path) in slots.items():
        record_path = output / "measurement" / ir_path / slot.split(":")[0]
        record_path.mkdir(parents=True, exist_ok=True)
        record_file = record_path / "record.json"
        resumed = _resume_record(record_file) if args.resume else None
        if resumed is not None:
            records[slot] = resumed
            continue
        command = [
            sys.executable,
            "-m",
            "scripts.internal.solar.l3a_worker",
            "measure",
            "--slot",
            slot,
            "--problem-dir",
            str(problem_dir),
            "--workload-uuid",
            workload,
            "--ir-path",
            ir_path,
            "--device",
            args.device,
            "--orojenesis-home",
            str(args.orojenesis_home),
            "--record",
            str(record_file),
        ]
        error: str | None = None
        try:
            _spawn(command, output / "worker-logs", slot, args.timeout)
        except subprocess.TimeoutExpired:
            error = f"worker timeout after {args.timeout}s"
        if error is None and not record_file.is_file():
            error = "worker produced no record"
        if error is not None:
            atomic_write_json_value(
                record_file,
                {
                    "schema_version": MEASUREMENT_SCHEMA,
                    "slot": slot,
                    "ir_path": ir_path,
                    "worker_error": error,
                    "stages": {},
                },
            )
        records[slot] = _load_json(record_file)
    cuda_mode = args.device.split(":", 1)[0] != "cpu"
    gate = evaluate_measurement_gate(records, list(slots), cuda_mode=cuda_mode)
    report = {
        "schema_version": MEASUREMENT_SCHEMA,
        "generated_at": utc_timestamp(),
        "identity": identity,
        "device": args.device,
        "denominator": sorted(slots),
        "gate": gate,
        "aggregates": {
            ir_path: aggregate_stages(
                {
                    slot: record
                    for slot, record in records.items()
                    if slot.endswith(f":{ir_path}")
                }
            )
            for ir_path in IR_PATH_VALUES
        },
    }
    atomic_write_json_value(
        output / "measurement" / "measurement-report.json", report
    )
    return report


def run_pilot_phase(
    output: Path, args: argparse.Namespace, identity: dict[str, Any]
) -> dict[str, Any]:
    """Run the eight make_fx pilot slots and the three focused samples."""
    slots: dict[str, Any] = {}
    crashed = False
    for item, path in R16_ITEMS.items():
        problem, workload = path.rsplit("/", 1)
        record_path = output / "fake-pilot" / item
        record_path.mkdir(parents=True, exist_ok=True)
        record_file = record_path / "pilot.json"
        resumed = _resume_record(record_file) if args.resume else None
        if resumed is not None:
            slots[item] = resumed
            continue
        command = [
            sys.executable,
            "-m",
            "scripts.internal.solar.l3a_worker",
            "pilot",
            "--slot",
            f"{item}:make_fx_aten",
            "--problem-dir",
            str(CORPUS_ROOT / problem),
            "--workload-uuid",
            workload,
            "--device",
            args.device,
            "--record",
            str(record_file),
        ]
        try:
            _spawn(
                command, output / "worker-logs", f"pilot-{item}", args.timeout
            )
        except subprocess.TimeoutExpired:
            atomic_write_json_value(
                record_file,
                {"slot": f"{item}:make_fx_aten", "worker_error": "timeout"},
            )
        if "worker_error" in _load_json(record_file):
            crashed = True
        slots[item] = _load_json(record_file)
    focused_dir = output / "fake-pilot" / "focused"
    focused_done = focused_dir / "focused.done"
    if not (args.resume and focused_done.is_file()):
        try:
            _spawn(
                [
                    sys.executable,
                    "-m",
                    "scripts.internal.solar.l3a_worker",
                    "focused",
                    "--device",
                    "cpu",
                    "--record-dir",
                    str(focused_dir),
                ],
                output / "worker-logs",
                "focused",
                args.timeout,
            )
        except subprocess.TimeoutExpired:
            crashed = True
        else:
            focused_done.write_text(utc_timestamp())
    focused = {
        path.stem: _load_json(path)
        for path in sorted(focused_dir.glob("*.json"))
    }
    gate = evaluate_pilot_gate(
        slots, focused, structure_check=_production_src_untouched()
    )
    summary = {
        "schema_version": PILOT_SCHEMA_SUMMARY,
        "generated_at": utc_timestamp(),
        "identity": identity,
        "decision_rule": (
            "docs/internal/solar-l3a-fake-static-pilot-acceptance-2026-11-05.md"
        ),
        "worker_crashed": crashed,
        "gate": gate,
        "slots": slots,
        "focused": focused,
    }
    atomic_write_json_value(
        output / "fake-pilot" / "pilot-summary.json", summary
    )
    return summary


def _production_src_untouched() -> bool:
    """Structural D4 check: only the new pilot module may touch src/."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "src"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    # LineB: the pilot module was adopted into the production backend and
    # deleted; src/ must now be completely untouched.
    return result.returncode == 0 and not lines


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    import torch

    properties = (
        torch.cuda.get_device_properties(args.device)
        if args.device.split(":", 1)[0] != "cpu" and torch.cuda.is_available()
        else None
    )
    raw_uuid = getattr(properties, "uuid", None)
    return {
        "git_commit": revision,
        "corpus_manifest_sha256": sha256_file(CORPUS_ROOT / "manifest.yaml"),
        "mapper_sha256": sha256_file(
            Path(args.orojenesis_home) / "bin" / "timeloop-mapper"
        ),
        "torch_version": torch.__version__,
        "gpu_name": getattr(properties, "name", None),
        "gpu_uuid": None if raw_uuid is None else str(raw_uuid),
    }


def main() -> int:
    """Run the requested campaign phases and print the gate verdicts."""
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
    parser.add_argument(
        "--phase",
        choices=("all", "measure", "pilot"),
        default="all",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    identity = _identity(args)
    measure_gate: dict[str, Any] | None = None
    pilot_gate: dict[str, Any] | None = None
    pilot_crashed = False
    if args.phase in {"all", "measure"}:
        report = run_measure_phase(args.output_root, args, identity)
        measure_gate = report["gate"]
    if args.phase in {"all", "pilot"}:
        summary = run_pilot_phase(args.output_root, args, identity)
        pilot_gate = summary["gate"]
        pilot_crashed = bool(summary["worker_crashed"])
    print(json.dumps({"measure": measure_gate, "pilot": pilot_gate}))
    measure_failed = bool(measure_gate and not measure_gate["passed"])
    return 1 if measure_failed or pilot_crashed else 0


if __name__ == "__main__":
    raise SystemExit(main())
