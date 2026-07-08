"""Compatibility facade for the relocated report module."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("sol_execbench.core.reports.evaluation_stability_builder")
