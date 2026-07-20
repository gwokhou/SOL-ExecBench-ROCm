# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable import surface used only by the trusted reference worker."""

from sol_execbench.core.bench.reference_service import serve_reference_worker

__all__ = ["serve_reference_worker"]
