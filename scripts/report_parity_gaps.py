#!/usr/bin/env python3
"""Compatibility wrapper for the internal parity gap report script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.internal.reports.report_parity_gaps import *  # noqa: F403
from scripts.internal.reports.report_parity_gaps import main


if __name__ == "__main__":
    main()
