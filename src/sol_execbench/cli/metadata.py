"""Compatibility facade for the relocated CLI module."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("sol_execbench.cli.commands.metadata")
