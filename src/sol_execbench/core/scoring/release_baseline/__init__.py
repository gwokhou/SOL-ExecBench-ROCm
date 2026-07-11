# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Release baseline evidence construction and verification APIs."""

from .models import (
    CLASSIFICATIONS,
    RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION,
    RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION,
    ReleaseBaselineBundle,
    ReleaseBaselineWorkload,
    ReleaseProvenance,
    load_release_baseline_bundle,
    release_baseline_bundle_from_dict,
    release_provenance_from_dict,
    sha256_file,
    write_release_baseline_bundle,
)
from .builder import (
    AuthorityInput,
    build_release_baseline_bundle,
    write_release_baseline_outputs,
)

__all__ = [
    "CLASSIFICATIONS",
    "AuthorityInput",
    "RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION",
    "RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION",
    "ReleaseBaselineBundle",
    "ReleaseBaselineWorkload",
    "ReleaseProvenance",
    "load_release_baseline_bundle",
    "release_baseline_bundle_from_dict",
    "release_provenance_from_dict",
    "sha256_file",
    "write_release_baseline_bundle",
    "build_release_baseline_bundle",
    "write_release_baseline_outputs",
]
