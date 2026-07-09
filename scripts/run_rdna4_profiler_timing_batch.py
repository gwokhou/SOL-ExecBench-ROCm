#!/usr/bin/env python3
"""Compatibility wrapper for the internal RDNA4 profiler timing batch script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.internal.rdna4 import (
    run_rdna4_profiler_timing_batch as _timing_batch,
)
from scripts.internal.rdna4.run_rdna4_profiler_timing_batch import *  # noqa: F403

_write_blocked_sidecar = _timing_batch._write_blocked_sidecar
main = _timing_batch.main

__all__ = [
    name
    for name in globals()
    if not name.startswith("_") and name not in {"Path", "sys"}
]
__all__.append("_write_blocked_sidecar")


if __name__ == "__main__":
    raise SystemExit(main())
