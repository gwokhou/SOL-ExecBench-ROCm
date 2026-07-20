# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable availability of publication-grade scoring authority."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.dataset.corpus import CorpusManifest


def official_score_availability(corpus_manifest: str | Path) -> dict[str, Any]:
    """Report the immutable corpus authority state without accepting score inputs."""
    corpus = CorpusManifest.load(corpus_manifest)
    scoring = corpus.official_scoring
    reason = str(scoring.get("reason_code") or "release_authority_not_published")
    required = [str(item) for item in scoring.get("required_evidence") or []]
    manifest_status = str(scoring.get("status"))
    return {
        "schema_version": "sol_execbench.official_score_availability.v1",
        "status": "unavailable",
        "reason_code": reason,
        "manifest_status": manifest_status,
        "required_evidence": required,
        "scorer_implemented": False,
        "accepts_caller_authored_inputs": False,
    }


__all__ = ["official_score_availability"]
