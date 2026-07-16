# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed Python access to AMD's machine-readable ISA specifications.

The package owns the Python API and invokes a small vendored C++ helper only
when callers explicitly open a specification.  Importing this module never
downloads data, compiles C++, or needs a ROCm GPU.
"""

from sol_execbench.tools.amd_isa.client import AmdIsa, open_isa
from sol_execbench.tools.amd_isa.errors import (
    IsaDecodeError,
    IsaDownloadError,
    IsaError,
    IsaHelperBuildError,
    IsaIntegrityError,
    IsaProtocolError,
    IsaSpecUnavailableError,
)
from sol_execbench.tools.amd_isa.repository import IsaSpecDescriptor, IsaSpecRepository

__all__ = [
    "AmdIsa",
    "IsaDecodeError",
    "IsaDownloadError",
    "IsaError",
    "IsaHelperBuildError",
    "IsaIntegrityError",
    "IsaProtocolError",
    "IsaSpecRepository",
    "IsaSpecDescriptor",
    "IsaSpecUnavailableError",
    "open_isa",
]
