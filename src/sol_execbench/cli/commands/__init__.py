"""Command helpers for the SOL ExecBench CLI."""

from importlib import import_module

_root = import_module("sol_execbench.cli.commands.root")

for _name in dir(_root):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_root, _name)

del _name, _root
