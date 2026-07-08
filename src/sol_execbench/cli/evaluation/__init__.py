"""Evaluation command helpers for the SOL ExecBench CLI."""

from sol_execbench.cli.evaluation import command as _command

for _name in dir(_command):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_command, _name)

del _name, _command
