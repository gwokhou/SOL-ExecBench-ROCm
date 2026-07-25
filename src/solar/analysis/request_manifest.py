# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical publication of one SOLAR request manifest."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import yaml

from solar.schema_versions import SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION


class RequestManifestView(Protocol):
    @property
    def analysis_id(self) -> str: ...

    @property
    def reference_name(self) -> str: ...

    @property
    def reference_sha256(self) -> str: ...

    @property
    def precision(self) -> str: ...

    @property
    def trace_seed(self) -> int: ...

    @property
    def verification_seeds(self) -> tuple[int, ...]: ...

    @property
    def atol(self) -> float: ...

    @property
    def rtol(self) -> float: ...

    @property
    def required_matched_ratio(self) -> float: ...

    @property
    def max_error_cap(self) -> float | None: ...

    @property
    def allow_negative_inf(self) -> bool: ...

    @property
    def require_orojenesis(self) -> bool: ...


class ArtifactManifestView(Protocol):
    @property
    def path(self) -> str: ...

    @property
    def sha256(self) -> str: ...


class BoundManifestView(Protocol):
    @property
    def seconds(self) -> float: ...

    @property
    def kind(self) -> str: ...

    @property
    def limiting_resource(self) -> str | None: ...


def write_request_manifest(
    request: RequestManifestView,
    staging: Path,
    architecture_sha256: str,
    artifacts: Sequence[ArtifactManifestView],
    bound: BoundManifestView,
    *,
    formal_bound_kind: str,
) -> None:
    """Write the content-addressed analysis contract and authority status."""
    manifest = {
        "schema_version": SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
        "analysis_id": request.analysis_id,
        "architecture_sha256": architecture_sha256,
        "reference": {
            "name": request.reference_name,
            "sha256": request.reference_sha256,
        },
        "analysis_contract": {
            "precision": request.precision,
            "trace_seed": request.trace_seed,
            "verification_seeds": list(request.verification_seeds),
            "atol": request.atol,
            "rtol": request.rtol,
            "required_matched_ratio": request.required_matched_ratio,
            "max_error_cap": request.max_error_cap,
            "allow_negative_inf": request.allow_negative_inf,
            "require_orojenesis": request.require_orojenesis,
        },
        "publication_eligible": bound.kind == formal_bound_kind,
        "artifacts": [
            {"path": artifact.path, "sha256": artifact.sha256} for artifact in artifacts
        ],
        "bound": {
            "seconds": bound.seconds,
            "kind": bound.kind,
            "limiting_resource": bound.limiting_resource,
        },
    }
    (staging / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))


__all__ = ["write_request_manifest"]
