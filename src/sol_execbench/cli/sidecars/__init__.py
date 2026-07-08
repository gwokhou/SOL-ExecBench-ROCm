"""Sidecar helpers for the SOL ExecBench CLI."""

from sol_execbench.cli.sidecars import common as _common

for _name in dir(_common):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_common, _name)

del _name, _common
