# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed verifier and scorer for publisher release bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from sol_execbench.core.dataset.aka_contract import AkaReleasePolicy
from sol_execbench.core.dataset.aka_corpus import AkaCorpusManifest
from sol_execbench.core.integrity import sha256_file, verify_artifact_file

from .aggregation import SuiteScore, WorkloadScore, aggregate_suite_scores
from .formula import sol_score
from .official_scoring import OFFICIAL_CORPUS_MANIFEST_SHA256
from .release_models import (
    BaselineStatement,
    CandidateStatement,
    MAX_RELEASE_STATEMENT_BYTES,
    ReleaseBundle,
    ReleaseModel,
    SolarIndexStatement,
)
from .release_solar import verify_solar_index
from .release_traces import VerifiedRun, verify_release_run

_Statement = TypeVar("_Statement", bound=ReleaseModel)


@dataclass(frozen=True, slots=True)
class OfficialScoreResult:
    """Publication-grade score plus its verified evidence identities."""

    candidate_id: str
    baseline_id: str
    source_revision: str
    release_bundle_sha256: str
    suite: SuiteScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "official",
            "candidate_id": self.candidate_id,
            "baseline_id": self.baseline_id,
            "source_revision": self.source_revision,
            "release_bundle_sha256": self.release_bundle_sha256,
            "score": self.suite.score,
            "problem_scores": self.suite.problem_scores,
            "scored_workloads": self.suite.scored_workloads,
        }


def verify_and_score_release(
    bundle_path: Path,
    *,
    corpus_manifest_path: Path,
) -> OfficialScoreResult:
    """Verify every content boundary and compute the official suite score."""
    bundle_file = bundle_path.resolve()
    bundle_root = bundle_file.parent
    bundle = _load_model(bundle_file, ReleaseBundle)
    corpus = AkaCorpusManifest.load(corpus_manifest_path)
    if sha256_file(corpus.path) != OFFICIAL_CORPUS_MANIFEST_SHA256:
        raise ValueError("official corpus manifest is not repository-pinned")
    _verify_bundle_corpus(bundle, bundle_root, corpus)
    baseline, candidate, solar = _load_statements(bundle, bundle_root)
    _verify_release_pins(corpus, baseline, candidate, solar)
    verified = _verify_runs(
        baseline,
        candidate,
        bundle_root=bundle_root,
        corpus=corpus,
    )
    baseline_run, candidate_run = verified
    bounds = verify_solar_index(solar, bundle_root=bundle_root, corpus=corpus)
    suite = _score_verified_runs(
        baseline_run,
        candidate_run,
        bounds,
    )
    return OfficialScoreResult(
        candidate_id=candidate.candidate_id,
        baseline_id=baseline.baseline_id,
        source_revision=baseline.source_revision,
        release_bundle_sha256=sha256_file(bundle_file),
        suite=suite,
    )


def _verify_bundle_corpus(
    bundle: ReleaseBundle,
    bundle_root: Path,
    corpus: AkaCorpusManifest,
) -> None:
    bundled = verify_artifact_file(
        bundle_root,
        bundle.corpus_manifest.path,
        expected_sha256=bundle.corpus_manifest.sha256,
        expected_size_bytes=bundle.corpus_manifest.size_bytes,
    )
    if sha256_file(bundled) != sha256_file(corpus.path):
        raise ValueError("release bundle corpus identity mismatch")


def _load_statements(
    bundle: ReleaseBundle,
    bundle_root: Path,
) -> tuple[
    BaselineStatement,
    CandidateStatement,
    SolarIndexStatement,
]:
    models = (
        (bundle.baseline, BaselineStatement),
        (bundle.candidate, CandidateStatement),
        (bundle.solar, SolarIndexStatement),
    )
    loaded: list[ReleaseModel] = []
    for reference, model in models:
        path = verify_artifact_file(
            bundle_root,
            reference.path,
            expected_sha256=reference.sha256,
            expected_size_bytes=reference.size_bytes,
        )
        loaded.append(_load_model(path, model))
    baseline, candidate, solar = loaded
    assert isinstance(baseline, BaselineStatement)
    assert isinstance(candidate, CandidateStatement)
    assert isinstance(solar, SolarIndexStatement)
    return baseline, candidate, solar


def _verify_release_pins(
    corpus: AkaCorpusManifest,
    baseline: BaselineStatement,
    candidate: CandidateStatement,
    solar: SolarIndexStatement,
) -> None:
    scoring = corpus.official_scoring
    if scoring.get("status") != "available":
        raise ValueError("corpus manifest does not authorize official scoring")
    if scoring.get("release_policy") != AkaReleasePolicy.CONTENT_ADDRESSED_PUBLISHER_V1:
        raise ValueError("corpus manifest uses an unsupported release policy")
    if scoring.get("baseline_id") != baseline.baseline_id:
        raise ValueError("official scoring baseline_id is not corpus-pinned")
    if not (
        baseline.source_revision == candidate.source_revision == solar.source_revision
    ):
        raise ValueError("release evidence source revisions do not match")


def _verify_runs(
    baseline: BaselineStatement,
    candidate: CandidateStatement,
    *,
    bundle_root: Path,
    corpus: AkaCorpusManifest,
) -> tuple[VerifiedRun, VerifiedRun]:
    verified = (
        verify_release_run(
            baseline,
            bundle_root=bundle_root,
            corpus=corpus,
            require_passed=True,
            require_reference_baseline=True,
        ),
        verify_release_run(
            candidate,
            bundle_root=bundle_root,
            corpus=corpus,
            require_passed=False,
        ),
    )
    if len({item.environment for item in verified}) != 1:
        raise ValueError("release runs use different environment identities")
    return verified


def _score_verified_runs(
    baseline: VerifiedRun,
    candidate: VerifiedRun,
    bounds: dict[tuple[str, str], float],
) -> SuiteScore:
    if (
        baseline.workloads.keys() != candidate.workloads.keys()
        or baseline.workloads.keys() != bounds.keys()
    ):
        raise ValueError("release evidence denominators differ")
    scores = [
        _score_workload(
            identity,
            baseline.workloads[identity].latency_ms,
            candidate.workloads[identity].latency_ms,
            candidate.workloads[identity].passed,
            bounds[identity],
        )
        for identity in sorted(bounds)
    ]
    return aggregate_suite_scores(scores)


def _score_workload(
    identity: tuple[str, str],
    baseline_latency: float | None,
    candidate_latency: float | None,
    candidate_passed: bool,
    sol_latency: float,
) -> WorkloadScore:
    assert baseline_latency is not None
    if not candidate_passed:
        return WorkloadScore(identity[0], identity[1], 0.0)
    if candidate_latency is None:
        raise ValueError(f"passing candidate has no latency for {identity}")
    score = sol_score(candidate_latency, baseline_latency, sol_latency)
    return WorkloadScore(identity[0], identity[1], score)


def _load_model(path: Path, model: type[_Statement]) -> _Statement:
    if not path.is_file() or path.stat().st_size > MAX_RELEASE_STATEMENT_BYTES:
        raise ValueError("release statement is missing or exceeds the size limit")
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid release statement: {path.name}") from exc


__all__ = ["OfficialScoreResult", "verify_and_score_release"]
