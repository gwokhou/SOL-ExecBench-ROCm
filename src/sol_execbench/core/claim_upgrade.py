"""Compatibility facade for the relocated report module."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("sol_execbench.core.reports.claim_upgrade")
