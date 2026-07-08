"""Compatibility facade for the relocated evidence module."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("sol_execbench.core.evidence.runtime_evidence")

if __name__ == "__main__":
    raise SystemExit(sys.modules[__name__].main())
