"""Compatibility facade for the relocated platform module."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("sol_execbench.core.platform.dependency_matrix")

if __name__ == "__main__":
    raise SystemExit(sys.modules[__name__].main())
