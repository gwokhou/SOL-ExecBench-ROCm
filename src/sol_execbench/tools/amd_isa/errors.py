# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable failures exposed by the AMD ISA tool layer."""


class ISAError(RuntimeError):
    """Base class for all project-owned AMD ISA tool failures."""


class ISASpecUnavailableError(ISAError):
    """A required local ISA specification is unavailable."""


class ISADownloadError(ISAError):
    """The pinned ISA archive could not be downloaded."""


class ISAIntegrityError(ISAError):
    """Downloaded ISA data did not satisfy the release lock."""


class ISAHelperBuildError(ISAError):
    """The vendored C++ JSON helper could not be built."""


class ISAProtocolError(ISAError):
    """The helper returned malformed or incompatible protocol output."""


class ISADecodeError(ISAError):
    """The loaded ISA specification could not decode a requested input."""
