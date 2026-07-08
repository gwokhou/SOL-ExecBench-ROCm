# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compatibility facade for the relocated CLI module."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("sol_execbench.cli.commands.baseline")
