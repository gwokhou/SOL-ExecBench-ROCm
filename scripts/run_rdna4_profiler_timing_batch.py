#!/usr/bin/env python3
"""Compatibility wrapper for the internal RDNA4 profiler timing batch script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.internal.rdna4.run_rdna4_profiler_timing_batch import *  # noqa: F403
from scripts.internal.rdna4.run_rdna4_profiler_timing_batch import (
    _write_blocked_sidecar,  # noqa: F401
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
