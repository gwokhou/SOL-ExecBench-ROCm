# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed Python access to AMD's machine-readable ISA specifications.

The package owns the Python API and invokes a small vendored C++ helper only
when callers explicitly open a specification.  Importing this module never
downloads data, compiles C++, or needs a ROCm GPU.
"""

from sol_execbench.tools.amd_isa.client import AMDIsa, open_isa
from sol_execbench.tools.amd_isa.errors import (
    ISADecodeError,
    ISADownloadError,
    ISAError,
    ISAHelperBuildError,
    ISAIntegrityError,
    ISAProtocolError,
    ISASpecUnavailableError,
)
from sol_execbench.tools.amd_isa.repository import (
    ISASpecDescriptor,
    ISASpecRepository,
)

__all__ = [
    "AMDIsa",
    "ISADecodeError",
    "ISADownloadError",
    "ISAError",
    "ISAHelperBuildError",
    "ISAIntegrityError",
    "ISAProtocolError",
    "ISASpecRepository",
    "ISASpecDescriptor",
    "ISASpecUnavailableError",
    "open_isa",
]
