"""Collect all frozen BASE-01 slots without GPU devices or production edits.

Run inside the pinned, network-isolated image; use a new output directory.
The summary records current behavior and is not a feature acceptance verdict.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any, TypedDict

from scripts.internal.solar.base01_observations import (
    AccountingStatus,
    Observation,
    ObservationDataError,
    ReceiptFailure,
    load_observation,
)
from scripts.internal.solar.base01_oracles import (
    GEOMETRY_BOUNDARIES,
    geometry_oracles,
    model_oracles,
)
from scripts.internal.solar.base01_records import (
    ROOT,
    environment_record,
    file_inventory,
    real_input_records,
    source_identity,
    write_record,
)
from scripts.internal.solar.base01_samples import SAMPLES, Sample
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from solar.contracts import SolarReadinessStatus
from solar.ir.contracts import IRPath

SLOT_TIMEOUT_SECONDS = 180
LOG_LIMIT_BYTES = 1_048_576
WORKER_MODULE = "scripts.internal.solar.base01_worker"


class ChildStatus(StrEnum):
    """Child process outcomes, separate from SOLAR semantic status."""

    EXITED = "exited"
    TIMEOUT = "timeout"


class SlotExecution(TypedDict, total=False):
    """Bounded process receipt for one frozen sample/path pair."""

    sample_id: str
    ir_path: str
    command: list[str]
    timeout_seconds: int
    log_tail_limit_bytes: int
    returncode: int | None
    status: ChildStatus
    observation: Observation | None
    receipt_failure: ReceiptFailure
    files: dict[str, str]


def collect_slot(sample: Sample, ir_path: IRPath, root: Path) -> SlotExecution:
    """Bound a child process and retain every planned slot, including crashes."""
    output = root / sample.sample_id / ir_path.value
    output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        WORKER_MODULE,
        "--sample",
        sample.sample_id,
        "--ir-path",
        ir_path.value,
        "--output",
        str(output),
    ]
    execution: SlotExecution = {
        "sample_id": sample.sample_id,
        "ir_path": ir_path.value,
        "command": command,
        "timeout_seconds": SLOT_TIMEOUT_SECONDS,
        "log_tail_limit_bytes": LOG_LIMIT_BYTES,
    }
    try:
        result = run_in_process_group_bounded(
            command,
            cwd=ROOT,
            timeout=SLOT_TIMEOUT_SECONDS,
            max_capture_bytes=LOG_LIMIT_BYTES,
        )
        execution.update(
            returncode=result.returncode, status=ChildStatus.EXITED
        )
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        execution.update(returncode=None, status=ChildStatus.TIMEOUT)
        stdout, stderr = str(exc.stdout or ""), str(exc.stderr or "")
    (output / "stdout.log").write_text(stdout or "")
    (output / "stderr.log").write_text(stderr or "")
    observed = output / "observation.json"
    execution["observation"] = None
    if execution["returncode"] == 0:
        try:
            execution["observation"] = load_observation(
                observed, sample, ir_path
            )
        except FileNotFoundError:
            execution["receipt_failure"] = ReceiptFailure.MISSING
        except ObservationDataError as exc:
            execution["receipt_failure"] = exc.reason
    execution["files"] = file_inventory(output)
    observation = execution["observation"]
    write_record(
        output / "execution.json",
        {
            **execution,
            "observation": observation.model_dump(
                mode="json", exclude_unset=True
            )
            if observation is not None
            else None,
        },
    )
    return execution


def summarize(slots: list[SlotExecution]) -> dict[str, Any]:
    """Report separate denominators and never convert failure to zero work."""
    semantic = 0
    accounting = 0
    failures: Counter[str] = Counter()
    mismatches = []
    for slot in slots:
        observed = slot["observation"]
        if observed is None:
            failures[
                "worker_" + str(slot.get("receipt_failure", slot["status"]))
            ] += 1
            continue
        ready = observed.readiness
        semantic += ready.status is SolarReadinessStatus.READY
        if ready.status is not SolarReadinessStatus.READY:
            failures[str(ready.reason_code)] += 1
        accounting += observed.accounting_status is AccountingStatus.COMPLETED
        for metric, value in observed.math_comparison.items():
            if not value.matches:
                mismatches.append(
                    {
                        "sample_id": slot["sample_id"],
                        "ir_path": slot["ir_path"],
                        "metric": metric,
                        **value.model_dump(),
                    }
                )
    return {
        "f_planned_slots": len(SAMPLES) * len(IRPath),
        "f_attempted_slots": len(slots),
        "f_semantic_passed": semantic,
        "f_accounting_completed": accounting,
        "f_failures": dict(failures),
        "math_mismatches": mismatches,
        "g_specs": 10,
        "m_specs": 10,
        "r_planned_slots": 16,
        "r_attempted_slots": 0,
        "complete_cache_aware": 0,
        "formal_qualified": 0,
        "gpu_experiments": 0,
    }


def main() -> int:
    """Freeze identities before observing current behavior, then seal evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    environment = environment_record()
    identity = source_identity()
    write_record(root / "environment.json", environment)
    write_record(root / "source-before.json", identity)
    write_record(root / "R-inputs.json", real_input_records())
    write_record(
        root / "G-oracles.json",
        {"cases": geometry_oracles(), "boundaries": GEOMETRY_BOUNDARIES},
    )
    write_record(root / "M-oracles.json", model_oracles())
    slots = []
    for sample in SAMPLES:
        for ir_path in IRPath:
            slot = collect_slot(sample, ir_path, root / "F")
            slots.append(slot)
            print(sample.sample_id, ir_path.value, slot["status"], flush=True)
    after = source_identity()
    write_record(root / "source-after.json", after)
    report = summarize(slots)
    report["source_unchanged_during_collection"] = identity == after
    report["artifacts"] = file_inventory(root)
    write_record(root / "summary.json", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "artifacts"},
            indent=2,
        )
    )
    return (
        0
        if identity == after
        and all(slot["observation"] is not None for slot in slots)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
