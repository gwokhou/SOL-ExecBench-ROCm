# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable failures exposed by the AMD ISA tool layer."""


class IsaError(RuntimeError):
    """Base class for all project-owned AMD ISA tool failures."""


class IsaSpecUnavailableError(IsaError):
    """A required local ISA specification is unavailable."""


class IsaDownloadError(IsaError):
    """The pinned ISA archive could not be downloaded."""


class IsaIntegrityError(IsaError):
    """Downloaded ISA data did not satisfy the release lock."""


class IsaHelperBuildError(IsaError):
    """The vendored C++ JSON helper could not be built."""


class IsaProtocolError(IsaError):
    """The helper returned malformed or incompatible protocol output."""


class IsaDecodeError(IsaError):
    """The loaded ISA specification could not decode a requested input."""
