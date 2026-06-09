#!/usr/bin/env python3
"""RDNA4 STABLE_PEAK clock-lock evidence collection with active GPU workload.

Records pre-state, locks clocks, runs sustained GPU compute, monitors
clock lock during the workload, then resets and records post-state.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from sol_execbench.core.bench.clock_lock import lock_clocks, unlock_clocks

OUTPUT_DIR = Path("out/rdna4-clock-lock-workload-20260609")


def rocm_smi(*args: str) -> str:
    result = subprocess.run(
        ["rocm-smi", *args],
        capture_output=True,
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
            clock_state = rocm_smi("--showclocks", "--showclkfrq")
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
    log(
        "pre_state",
        rocm_smi(
            "--showclocks",
            "--showperflevel",
            "--showproductname",
            "--showdriverversion",
            "--showhw",
        ),
    )
    log("pre_power_temp", rocm_smi("--showpower", "--showtemp", "--showuse"))

    # 2. Lock with the production clock-lock helper.
    if not lock_clocks():
        log("lock_failed", "STABLE_PEAK lock failed\n")
        raise SystemExit(1)
    log(
        "after_lock",
        rocm_smi(
            "--showclocks",
            "--showperflevel",
            "--showclkfrq",
            "--showproductname",
            "--showdriverversion",
            "--showhw",
        ),
    )
    log("after_lock_power", rocm_smi("--showpower", "--showtemp", "--showuse"))

    # 3. Run sustained GPU workload while monitoring clocks
    print("--- Running GPU workload with clock monitoring ---")
    run_gpu_workload(duration_sec=20.0)

    # 4. Post-workload state (with clocks still locked)
    log(
        "post_workload",
        rocm_smi(
            "--showclocks",
            "--showperflevel",
            "--showclkfrq",
        ),
    )
    log("post_workload_power", rocm_smi("--showpower", "--showtemp", "--showuse"))

    # 5. Reset clocks through the production helper.
    unlock_clocks()
    time.sleep(1)
    log(
        "after_reset",
        rocm_smi(
            "--showclocks",
            "--showperflevel",
            "--showclkfrq",
            "--showproductname",
            "--showdriverversion",
            "--showhw",
        ),
    )
    log("after_reset_power", rocm_smi("--showpower", "--showtemp", "--showuse"))

    print("=" * 60)
    print("Clock-lock evidence collection complete.")
    print(f"All logs in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
