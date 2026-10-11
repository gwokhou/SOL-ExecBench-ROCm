# SPDX-License-Identifier: Apache-2.0
"""Trusted single-case worker shared by reconstructed and upgraded containers.

Keep syntax compatible with Python 3.12. Qualification and process timeouts are
owned by the launcher. This worker is not an untrusted candidate evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from sol_execbench.core.bench.gpu_lock import acquire_gpu_lock
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.data.workload import NumericCheck, Workload
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

WARMUPS = 10
ITERATIONS = 100
MINIMUM_FREE_FACTOR = 8
EXPECTED_ARCHITECTURE = "gfx1200"
AUTO_LEVELS = frozenset(("AUTO", "AMDSMI_DEV_PERF_LEVEL_AUTO"))


def _clock_state() -> dict:
    result = run_in_process_group_bounded(
        ["amd-smi", "metric", "--gpu", "0", "--perf-level", "--json"],
        timeout=15,
        max_capture_bytes=8192,
    )
    if result.returncode:
        raise RuntimeError("AMD SMI performance-level query failed")
    payload = json.loads(result.stdout)
    rows = payload.get("gpu_data", []) if isinstance(payload, dict) else payload
    if len(rows) != 1 or rows[0].get("perf_level") not in AUTO_LEVELS:
        raise ValueError("expected one GPU at observed AUTO performance level")
    return rows[0]


def _device_state(item: dict) -> dict:
    import torch

    if torch.cuda.device_count() != 1:
        raise ValueError("exactly one visible GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if properties.gcnArchName.split(":")[0] != EXPECTED_ARCHITECTURE:
        raise ValueError("GPU architecture drift")
    decoded = bytes.fromhex(str(properties.uuid).replace("-", "")).decode(
        "ascii"
    )
    expected = os.environ["ROCR_VISIBLE_DEVICES"].removeprefix("GPU-")
    bdf = os.environ["UPGRADE_GPU_BDF"]
    device_id = (
        (Path("/sys/bus/pci/devices") / bdf / "unique_id").read_text().strip()
    )
    if decoded != expected or device_id != expected:
        raise ValueError("GPU UUID/BDF mismatch")
    free, total = torch.cuda.mem_get_info()
    if free < item["minimum_tensor_storage_bytes"] * MINIMUM_FREE_FACTOR:
        raise ValueError("insufficient free memory for conservative preflight")
    return {
        "free_bytes_before": free,
        "total_bytes": total,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu_name": properties.name,
        "architecture": properties.gcnArchName,
        "torch_uuid": str(properties.uuid),
        "bdf": bdf,
        "driver_kernel": os.uname().release,
        "rocr_visible_devices": os.environ["ROCR_VISIBLE_DEVICES"],
    }


def _inputs(definition: dict, workload: dict, path: Path) -> tuple[dict, dict]:
    import torch
    from safetensors.torch import load_file

    tensors = load_file(path, device="cpu")
    arguments, checksums = {}, {}
    for name, spec in definition["inputs"].items():
        descriptor = workload["inputs"][name]
        if descriptor["type"] == "scalar":
            arguments[name] = descriptor["value"]
            continue
        if (
            descriptor["type"] != "safetensors"
            or descriptor["path"] != path.name
            or descriptor["tensor_key"] != name
        ):
            raise ValueError(
                "screening worker requires exact frozen safetensors/scalar inputs"
            )
        shape = [
            workload["axes"][axis] if axis in workload["axes"] else int(axis)
            for axis in spec["shape"]
        ]
        tensor = tensors[name]
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != getattr(
            torch, spec["dtype"]
        ):
            raise ValueError("frozen tensor shape/dtype mismatch")
        checksums[name] = hashlib.sha256(
            tensor.view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        arguments[name] = tensor.to("cuda")
    return arguments, checksums


def _hot_latency(function: Callable, arguments: dict) -> float:
    import torch

    for _ in range(WARMUPS):
        function(**arguments)
    torch.cuda.synchronize()
    samples = []
    for _ in range(ITERATIONS):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        function(**arguments)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def measure(root: Path, item: dict, *, timed: bool) -> dict:
    """Compare compiled output to eager output before any hot timing."""
    import torch

    from sol_execbench.core.bench.correctness import compute_error_stats
    from sol_execbench.core.dataset.aka_equivalence import execute_reference

    definition = load_json_value(root / item["definition"])
    workload = load_json_value(root / item["workload"])
    Workload.model_validate(workload)
    candidate = load_json_value(root / item["candidate"])
    if sha256_file(root / item["candidate"]) != item["candidate_sha256"]:
        raise ValueError("candidate identity drift")
    device_state = _device_state(item)
    clocks_before = _clock_state() if timed else None
    tensor_path = root / item["tensors"]
    if sha256_file(tensor_path) != item["tensors_sha256"]:
        raise ValueError("frozen tensor identity drift")
    arguments, checksums = _inputs(definition, workload, tensor_path)
    reference = execute_reference(definition["reference"])
    expected = reference(**arguments)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    function = execute_reference(candidate["sources"][0]["content"])
    actual = function(**arguments)
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - started) * 1000
    check = NumericCheck.model_validate(workload["checks"][0])
    correctness, exceeds = compute_error_stats(actual, expected, check)
    correct = (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and not exceeds
    )
    return {
        "id": item["id"],
        "correct": correct,
        "input_sha256": checksums,
        "candidate_sha256": item["candidate_sha256"],
        "cold_compile_ms": cold_ms,
        "hot_execution_ms": _hot_latency(function, arguments)
        if correct and timed
        else None,
        "correctness": correctness.model_dump(mode="json"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        **device_state,
        "clocks_before": clocks_before,
        "clocks_after": _clock_state() if timed else None,
        "timed": timed,
    }


def main() -> None:
    """Use fresh compiler caches per process and preserve numerical failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--item", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timed", action="store_true")
    args = parser.parse_args()
    manifest = load_json_value(args.root / "manifest.json")
    item = next(item for item in manifest["items"] if item["id"] == args.item)
    with tempfile.TemporaryDirectory(prefix="upgrade-compiler-") as cache:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(cache) / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(Path(cache) / "triton")
        with acquire_gpu_lock(0, timeout_seconds=30):
            result = measure(args.root, item, timed=args.timed)
    atomic_write_json_value(args.output, result)
    args.output.chmod(0o644)
    if not result["correct"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
