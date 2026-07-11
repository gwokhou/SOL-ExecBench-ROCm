# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Release-scoped scoring baseline evidence contracts."""

from .models import (
    CLASSIFICATIONS,
    RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION,
    RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION,
    ReleaseBaselineBundle,
    ReleaseBaselineVerification,
    ReleaseBaselineVerificationWorkload,
    ReleaseBaselineWorkload,
    ReleaseProvenance,
    load_release_baseline_bundle,
    release_baseline_bundle_from_dict,
    release_baseline_verification_from_dict,
    sha256_file,
    write_release_baseline_bundle,
    write_release_baseline_verification,
)
from .builder import (
    AuthorityInput,
    build_release_baseline_bundle,
    write_release_baseline_outputs,
)
from .verifier import verify_release_baseline_rerun
from .authority import OfficialReleaseBaseline, load_official_release_baseline

__all__ = [
    "CLASSIFICATIONS",
    "AuthorityInput",
    "OfficialReleaseBaseline",
    "RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION",
    "RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION",
    "ReleaseBaselineBundle",
    "ReleaseBaselineVerification",
    "ReleaseBaselineVerificationWorkload",
    "ReleaseBaselineWorkload",
    "ReleaseProvenance",
    "build_release_baseline_bundle",
    "load_release_baseline_bundle",
    "load_official_release_baseline",
    "release_baseline_bundle_from_dict",
    "release_baseline_verification_from_dict",
    "sha256_file",
    "write_release_baseline_bundle",
    "write_release_baseline_verification",
    "write_release_baseline_outputs",
    "verify_release_baseline_rerun",
]
