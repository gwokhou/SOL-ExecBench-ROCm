#!/usr/bin/env python3
"""Compatibility wrapper for the internal consistency report script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.internal.reports.report_consistency import *  # noqa: F403
from scripts.internal.reports.report_consistency import main


if __name__ == "__main__":
    raise SystemExit(main())
