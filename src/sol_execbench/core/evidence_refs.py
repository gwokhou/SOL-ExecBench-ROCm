"""Compatibility facade for the relocated evidence module."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("sol_execbench.core.evidence.evidence_refs")
