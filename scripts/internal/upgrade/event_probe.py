# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Trusted, bounded HIP event fixture; never a submitted candidate kernel."""

import json
import time

import torch

from sol_execbench.core.bench.gpu_lock import acquire_gpu_lock

ELEMENTS = 1024
LOCK_TIMEOUT_SECONDS = 30
PRODUCER_DELAY_CYCLES = 50_000_000


def main() -> None:
    """Exercise record/wait, fan-out and handle reuse with checked results."""
    with acquire_gpu_lock(0, timeout_seconds=LOCK_TIMEOUT_SECONDS):
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        if properties.gcnArchName.split(":")[0] != "gfx1200":
            raise RuntimeError("upgrade probe requires gfx1200")
        source = torch.ones(ELEMENTS, device="cuda")
        torch.cuda.synchronize()
        producer, consumer, fanout = (torch.cuda.Stream() for _ in range(3))
        event = torch.cuda.Event()
        outputs = []
        started = time.perf_counter()
        for generation in range(2):
            with torch.cuda.stream(producer):
                value = source + generation
                torch.cuda._sleep(PRODUCER_DELAY_CYCLES)
                event.record()
            with torch.cuda.stream(consumer):
                consumer.wait_event(event)
                outputs.append(value * 2)
            with torch.cuda.stream(fanout):
                fanout.wait_event(event)
                outputs.append(value * 3)
            torch.cuda.synchronize()
        workload_ms = (time.perf_counter() - started) * 1000
        for output, expected in zip(outputs, (2, 3, 4, 6), strict=True):
            torch.testing.assert_close(
                output, torch.full_like(output, expected)
            )
        print(
            json.dumps(
                {
                    "correct": True,
                    "generations": 2,
                    "outputs": 4,
                    "workload_wall_ms": workload_ms,
                }
            )
        )


if __name__ == "__main__":
    main()
