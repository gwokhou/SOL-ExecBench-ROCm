# SPDX-License-Identifier: Apache-2.0
"""Serial Docker acquisition behind the complete upgrade qualification chain."""

import argparse
import re
import time
import uuid
from pathlib import Path

from scripts.internal.upgrade.paired_analysis import (
    BLOCKS,
    MeasurementKind,
    PairedMeasurement,
    analyze,
)
from scripts.internal.upgrade.qualification import (
    WORKER,
    identities,
    require_gate,
    validate_inputs,
)
from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationStage,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

CASE_TIMEOUT_SECONDS = 300
TOTAL_TIMEOUT_SECONDS = 3 * 60 * 60
IMAGE_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _command(
    protocol: dict,
    root: Path,
    item: dict,
    stack: str,
    output: Path,
    *,
    timed: bool,
    name: str,
) -> list[str]:
    image = protocol["images"][stack]
    if IMAGE_PATTERN.fullmatch(image) is None:
        raise ValueError("immutable image ID required")
    arguments = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        name,
        "--network=none",
        "--user=0:0",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--mount",
        f"type=bind,src={root},dst=/frozen,readonly",
        "--mount",
        f"type=bind,src={WORKER.resolve()},dst=/case_worker.py,readonly",
        "--mount",
        f"type=bind,src={output.parent},dst=/results",
        "--mount",
        "type=bind,src=/tmp/sol-execbench-locks,dst=/tmp/sol-execbench-locks",
        "-e",
        f"ROCR_VISIBLE_DEVICES={protocol['gpu_uuid']}",
        "-e",
        f"UPGRADE_GPU_BDF={protocol['gpu_bdf']}",
        "-e",
        "PYTORCH_ROCM_ARCH=gfx1200",
        "--entrypoint",
        "/venv/bin/python",
        image,
        "/case_worker.py",
        "--root",
        "/frozen",
        "--item",
        item["id"],
        "--output",
        f"/results/{output.name}",
    ]
    if timed:
        arguments.append("--timed")
    return arguments


def execute(
    protocol: dict,
    root: Path,
    item: dict,
    stack: str,
    output: Path,
    *,
    timed: bool,
) -> dict:
    """Run only an owned container; keep failures and terminate it on timeout."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"refusing to overwrite observation: {output}")
    name = f"sol-upgrade-{uuid.uuid4().hex}"
    try:
        result = run_in_process_group_bounded(
            _command(
                protocol, root, item, stack, output, timed=timed, name=name
            ),
            timeout=CASE_TIMEOUT_SECONDS,
            max_capture_bytes=16384,
        )
    finally:
        run_in_process_group_bounded(
            ["docker", "rm", "--force", name],
            timeout=30,
            max_capture_bytes=1024,
        )
    output.with_suffix(".log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(
            f"{stack}/{item['id']} failed: {output.with_suffix('.log')}"
        )
    row = load_json_value(output)
    if (
        row["id"] != item["id"]
        or row["correct"] is not True
        or row["candidate_sha256"] != item["candidate_sha256"]
        or row["torch"] != protocol["torch_versions"][stack]
        or row["rocr_visible_devices"] != protocol["gpu_uuid"]
        or row["timed"] is not timed
    ):
        raise ValueError("worker identity or correctness mismatch")
    row.update(stack=stack, image=protocol["images"][stack])
    atomic_write_json_value(output, row)
    print(f"{stack} {item['id']}: passed", flush=True)
    return row


def _pair(
    protocol: dict,
    item: dict,
    block: int,
    results: dict,
    protocol_sha256: str,
    kind: MeasurementKind,
) -> PairedMeasurement:
    old, new = results["old"], results["new"]
    if old["input_sha256"] != new["input_sha256"]:
        raise ValueError("old/new inputs differ; whole comparison rejected")
    return PairedMeasurement(
        workload=item["id"],
        block=block,
        candidate_sha256=item["candidate_sha256"],
        protocol_sha256=protocol_sha256,
        measurement_kind=kind,
        gpu_uuid=protocol["gpu_uuid"],
        old_image=protocol["images"]["old"],
        new_image=protocol["images"]["new"],
        old_first=block % 2 == 0,
        correct=True,
        old_ms=old[f"{kind}_ms"],
        new_ms=new[f"{kind}_ms"],
    )


def run(root: Path, plan: Path, qualification: Path) -> None:
    """Collect ten balanced independent pairs without dropping failed cases."""
    protocol = load_json_value(plan)
    manifest = validate_inputs(root)
    expected = identities(root, plan)
    output = Path(protocol["measurements_root"])
    output.mkdir(parents=True, exist_ok=False)
    workload_ids = [i["id"] for i in manifest["items"]]
    rows = {kind: [] for kind in MeasurementKind}
    deadline = time.monotonic() + TOTAL_TIMEOUT_SECONDS
    for block in range(BLOCKS):
        require_gate(root, plan, qualification, BatchGPUQualificationStage.FULL)
        order = ("old", "new") if block % 2 == 0 else ("new", "old")
        for index, item in enumerate(manifest["items"]):
            if (
                identities(root, plan) != expected
                or time.monotonic() >= deadline
            ):
                raise ValueError("source drift or acquisition budget exhausted")
            results = {
                stack: execute(
                    protocol,
                    root,
                    item,
                    stack,
                    output / f"block-{block}" / f"{index}-{stack}.json",
                    timed=True,
                )
                for stack in order
            }
            for kind in MeasurementKind:
                rows[kind].append(
                    _pair(
                        protocol, item, block, results, sha256_file(plan), kind
                    )
                )
                atomic_write_json_value(
                    output / f"{kind}-partial.json",
                    {
                        "workload_ids": workload_ids,
                        "pairs": [
                            row.model_dump(mode="json") for row in rows[kind]
                        ],
                    },
                )
    for kind in MeasurementKind:
        atomic_write_json_value(
            output / f"{kind}-report.json", analyze(rows[kind], workload_ids)
        )


def main() -> None:
    """Acquire only after exact static, canary and full gates exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.inputs.resolve(),
        args.plan.resolve(),
        args.qualification_root.resolve(),
    )


if __name__ == "__main__":
    main()
