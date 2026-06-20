#!/usr/bin/env python3
"""Compatibility wrapper for the internal Matrix diff report script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.internal.reports.diff_matrix_reports import *  # noqa: F403
from scripts.internal.reports.diff_matrix_reports import main


if __name__ == "__main__":
    raise SystemExit(main())
