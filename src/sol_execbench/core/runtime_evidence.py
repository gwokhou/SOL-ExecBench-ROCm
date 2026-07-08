"""Compatibility facade for the relocated evidence module."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("sol_execbench.core.evidence.runtime_evidence")

if __name__ == "__main__":
    raise SystemExit(_sys.modules[__name__].main())
