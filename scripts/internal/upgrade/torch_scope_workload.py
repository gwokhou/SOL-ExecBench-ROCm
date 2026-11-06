# SPDX-License-Identifier: Apache-2.0
"""Trusted eager PyTorch executor for opt-in dispatch scope acceptance."""

import argparse
import ctypes
import json
import os
from collections.abc import Callable
from pathlib import Path

import torch

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity.checksums import sha256_file

PROBLEMS = {
    "gemm": "l1n8_matmul_irregular",
    "composite": "3267_doubled_matmul",
    "event_chain": "3267_doubled_matmul",
    "copy_lane": "l1n8_matmul_irregular",
    "copy_event": "3267_doubled_matmul",
    "default_mix": "l1n8_matmul_irregular",
}
SHAPE = (513, 1025, 255)
EVENT_SHAPE = (4096, 4096, 4096)
WARMUP = 2
PINNED_CASES = frozenset({"copy_lane", "copy_event"})


def _scoped(
    case: str,
    a: torch.Tensor,
    b: torch.Tensor,
    a_host: torch.Tensor,
    sink_host: torch.Tensor,
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    producer: torch.cuda.Stream,
    consumer: torch.cuda.Stream,
    event: torch.cuda.Event,
) -> torch.Tensor:
    """Scoped body only; every case keeps all submissions on intended lanes."""
    if case == "copy_lane":
        with torch.cuda.stream(producer):
            dev = torch.empty_like(a)
            dev.copy_(a_host, non_blocking=True)
            result = function(dev, b)
            sink_host.copy_(result, non_blocking=True)
        return result
    if case == "copy_event":
        with torch.cuda.stream(producer):
            # Repeated producer work keeps the queue busy well past the
            # host-side WAIT submission, so a real GPU barrier is observable.
            result = None
            for _ in range(8):
                result = function(a, b)
            dev = torch.empty_like(a)
            # The engine copy stays the last pre-record submission, so the
            # verified event endpoint resolves to the copy.
            dev.copy_(a_host, non_blocking=True)
            event.record(producer)
        with torch.cuda.stream(consumer):
            consumer.wait_event(event)
            return result * 2
    if case == "default_mix":
        defaulted = function(a, b)
        with torch.cuda.stream(producer):
            return defaulted * 2
    with torch.cuda.stream(producer):
        copied = torch.empty_like(a)
        copied.copy_(a)
        result = function(copied, b)
        if case == "event_chain":
            event.record(producer)
    if case == "event_chain":
        with torch.cuda.stream(consumer):
            consumer.wait_event(event)
            result = result * 2
    return result


def execute(case: str, tool: Path, output: Path, run_id: str) -> None:
    """Keep all initialization, warmup and correctness work outside the scope."""
    repo = Path(__file__).resolve().parents[3]
    definition = (
        repo / "problems/AMD_AKA/torch2hip" / PROBLEMS[case] / "definition.json"
    )
    source = json.loads(definition.read_text())["reference"]
    namespace = {}
    exec(compile(source, str(definition), "exec"), namespace)  # noqa: S102 - audited repository reference
    function = namespace["run"]
    torch.manual_seed(20261011)
    dtype = (
        torch.float16
        if case in ("gemm", "copy_lane", "default_mix")
        else (torch.float32)
    )
    shape = EVENT_SHAPE if case in ("event_chain", "copy_event") else SHAPE
    m, k, n = shape
    a_cpu = torch.randn(m, k, dtype=dtype)
    b_cpu = torch.randn(k, n, dtype=dtype)
    expected = function(a_cpu, b_cpu)
    a, b = a_cpu.cuda(), b_cpu.cuda()
    a_host = a_cpu.pin_memory() if case in PINNED_CASES else a_cpu
    sink_host = (
        torch.empty(m, n, dtype=dtype, pin_memory=True)
        if case == "copy_lane"
        else a_cpu
    )
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    event = torch.cuda.Event()
    for _ in range(WARMUP):
        function(a, b)
    library = ctypes.CDLL(str(tool))
    boundary = library.sol_scope_boundary
    boundary.argtypes = [ctypes.c_int]
    boundary.restype = None
    torch.cuda.synchronize()
    boundary(1)
    result = _scoped(
        case, a, b, a_host, sink_host, function, producer, consumer, event
    )
    torch.cuda.synchronize()
    boundary(0)
    if case in ("event_chain", "copy_event", "default_mix"):
        expected = expected * 2
    torch.testing.assert_close(result.cpu(), expected, rtol=0.01, atol=0.02)
    device_uuid = str(torch.cuda.get_device_properties(0).uuid)
    torch.cuda.synchronize()
    seal = library.sol_scope_seal
    seal.argtypes = []
    seal.restype = None
    seal()
    atomic_write_json_value(
        output,
        {
            "run_id": run_id,
            "workload_id": case,
            "process_id": os.getpid(),
            "definition_sha256": sha256_file(definition),
            "executor_sha256": sha256_file(Path(__file__)),
            "collector_sha256": sha256_file(tool),
            "device_uuid": device_uuid,
            "shape": list(shape),
            "correct": True,
            "completed": True,
        },
    )


def main() -> None:
    """Run only under the separate acceptance controller and host clock guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=PROBLEMS)
    parser.add_argument("tool", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("run_id")
    args = parser.parse_args()
    execute(args.case, args.tool, args.output, args.run_id)


if __name__ == "__main__":
    main()
