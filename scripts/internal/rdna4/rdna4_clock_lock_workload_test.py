#!/usr/bin/env python3
"""RDNA4 STABLE_PEAK clock-lock evidence collection with active GPU workload.

Records pre-state, locks clocks, runs sustained GPU compute, monitors
clock lock during the workload, then resets and records post-state.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from sol_execbench.core.bench.clock_lock import acquire_clock_lock

OUTPUT_DIR = Path("out/rdna4-clock-lock-workload-20260609")


def amd_smi(*args: str) -> str:
    executable = shutil.which("amd-smi") or "/opt/rocm/bin/amd-smi"
    result = subprocess.run(
        [executable, *args],
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    return result.stdout


def log(label: str, content: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = label.replace(" ", "_").replace("/", "_")
    path = OUTPUT_DIR / f"{safe}.txt"
    path.write_text(content)
    print(f"[{label}]")
    for line in content.strip().splitlines():
        print(f"  {line}")
    print()


def run_gpu_workload(duration_sec: float = 45.0) -> None:
    import torch

    end = time.monotonic() + duration_sec
    i = 0
    log("gpu_workload_start", f"Starting GEMM workload for {duration_sec}s\n")
    while time.monotonic() < end:
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        _ = a @ b
        torch.cuda.synchronize()
        i += 1
        if i % 5 == 0:
            elapsed = time.monotonic()
            remaining = max(0, end - elapsed)
            clock_state = amd_smi("metric", "-c", "-l")
            log(f"clock_check_at_{int(elapsed)}s", clock_state)
            print(f"  ... {i} iterations done, {remaining:.1f}s remaining")
    # Final sync
    torch.cuda.synchronize()
    log("gpu_workload_end", f"Completed {i} GEMM iterations in {duration_sec}s\n")


def main() -> None:
    print("=" * 60)
    print("RDNA4 STABLE_PEAK Clock-Lock Workload Test")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)
    print()

    # 1. Pre-state (read-only)
    log("pre_static", amd_smi("static", "-a"))
    log("pre_metrics", amd_smi("metric", "-c", "-l", "-p", "-t", "-u"))

    # 2. Lock with the production clock-lock helper.
    clock_lock = acquire_clock_lock()
    if not clock_lock.locked:
        log("lock_failed", "STABLE_PEAK lock failed\n")
        raise SystemExit(1)
    with clock_lock:
        log("after_lock", amd_smi("metric", "-c", "-l", "-p", "-t", "-u"))

        # 3. Run sustained GPU workload while monitoring clocks
        print("--- Running GPU workload with clock monitoring ---")
        run_gpu_workload(duration_sec=20.0)

        # 4. Post-workload state (with clocks still locked)
        log(
            "post_workload",
            amd_smi("metric", "-c", "-l", "-p", "-t", "-u"),
        )
    # 5. The lease resets owned clocks and preserves an external lock.
    time.sleep(1)
    log("after_release", amd_smi("metric", "-c", "-l", "-p", "-t", "-u"))

    print("=" * 60)
    print("Clock-lock evidence collection complete.")
    print(f"All logs in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
