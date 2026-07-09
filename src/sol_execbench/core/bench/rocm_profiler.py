# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compatibility facade for ROCm profiler public API."""

import sol_execbench.core.bench.rocm_profiler_api as _rocm_profiler_api
from sol_execbench.core.bench.rocm_profiler_api import *  # noqa: F403

__all__ = _rocm_profiler_api.__all__
