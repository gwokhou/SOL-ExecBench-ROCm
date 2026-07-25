# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed verifier and scorer for signed release evidence bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from sol_execbench.core.dataset.aka_corpus import AkaCorpusManifest
from sol_execbench.core.integrity import sha256_file, verify_artifact_file

from .aggregation import SuiteScore, WorkloadScore, aggregate_suite_scores
from .formula import sol_score
from .official_authority import OFFICIAL_CORPUS_MANIFEST_SHA256
from .release_authority import (
    ReleaseAuthority,
    load_release_authority,
    verify_signed_statement,
)
from .release_models import (
    BaselineStatement,
    CandidateStatement,
    MAX_SIGNED_STATEMENT_BYTES,
    ReleaseBundle,
    ReleaseModel,
    RerunStatement,
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
    """Verify every authority boundary and compute the official suite score."""
    bundle_file = bundle_path.resolve()
    bundle_root = bundle_file.parent
    bundle = _load_model(bundle_file, ReleaseBundle)
    corpus = AkaCorpusManifest.load(corpus_manifest_path)
    if sha256_file(corpus.path) != OFFICIAL_CORPUS_MANIFEST_SHA256:
        raise ValueError("official corpus manifest is not repository-pinned")
    _verify_bundle_corpus(bundle, bundle_root, corpus)
    authority = load_release_authority(corpus.path)
    statements = _load_signed_statements(bundle, bundle_root, corpus, authority)
    baseline, rerun, candidate, solar = statements
    _verify_release_pins(corpus, bundle, baseline, rerun, solar)
    verified = _verify_runs(
        baseline,
        rerun,
        candidate,
        bundle_root=bundle_root,
        corpus=corpus,
    )
    baseline_run, rerun_run, candidate_run = verified
    bounds = verify_solar_index(solar, bundle_root=bundle_root, corpus=corpus)
    suite = _score_verified_runs(
        baseline_run,
        rerun_run,
        candidate_run,
        bounds,
        max_rerun_delta=authority.max_rerun_relative_delta,
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


def _load_signed_statements(
    bundle: ReleaseBundle,
    bundle_root: Path,
    corpus: AkaCorpusManifest,
    authority: ReleaseAuthority,
) -> tuple[
    BaselineStatement,
    RerunStatement,
    CandidateStatement,
    SolarIndexStatement,
]:
    models = (
        (bundle.baseline, BaselineStatement),
        (bundle.rerun, RerunStatement),
        (bundle.candidate, CandidateStatement),
        (bundle.solar, SolarIndexStatement),
    )
    loaded: list[ReleaseModel] = []
    for signed, model in models:
        path = verify_signed_statement(
            signed,
            bundle_root=bundle_root,
            corpus_root=corpus.authored_root,
            authority=authority,
        )
        loaded.append(_load_model(path, model))
    baseline, rerun, candidate, solar = loaded
    assert isinstance(baseline, BaselineStatement)
    assert isinstance(rerun, RerunStatement)
    assert isinstance(candidate, CandidateStatement)
    assert isinstance(solar, SolarIndexStatement)
    return baseline, rerun, candidate, solar


def _verify_release_pins(
    corpus: AkaCorpusManifest,
    bundle: ReleaseBundle,
    baseline: BaselineStatement,
    rerun: RerunStatement,
    solar: SolarIndexStatement,
) -> None:
    scoring = corpus.official_scoring
    if scoring.get("status") != "available":
        raise ValueError("corpus manifest does not authorize official scoring")
    if scoring.get("baseline_id") != baseline.baseline_id:
        raise ValueError("official scoring baseline_id is not corpus-pinned")
    if rerun.baseline_payload_sha256 != bundle.baseline.payload.sha256 or not (
        baseline.source_revision == rerun.source_revision == solar.source_revision
    ):
        raise ValueError("release evidence source revisions do not match")
    baseline_traces = {
        problem.problem_path: problem.trace.sha256 for problem in baseline.problems
    }
    rerun_traces = {
        problem.problem_path: problem.trace.sha256 for problem in rerun.problems
    }
    if any(
        rerun_traces.get(problem_path) == trace_sha256
        for problem_path, trace_sha256 in baseline_traces.items()
    ):
        raise ValueError("independent rerun reuses a baseline trace artifact")


def _verify_runs(
    baseline: BaselineStatement,
    rerun: RerunStatement,
    candidate: CandidateStatement,
    *,
    bundle_root: Path,
    corpus: AkaCorpusManifest,
) -> tuple[VerifiedRun, VerifiedRun, VerifiedRun]:
    if candidate.source_revision != baseline.source_revision:
        raise ValueError("candidate source revision is outside the release")
    verified = (
        verify_release_run(
            baseline,
            bundle_root=bundle_root,
            corpus=corpus,
            require_passed=True,
        ),
        verify_release_run(
            rerun,
            bundle_root=bundle_root,
            corpus=corpus,
            require_passed=True,
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
    rerun: VerifiedRun,
    candidate: VerifiedRun,
    bounds: dict[tuple[str, str], float],
    *,
    max_rerun_delta: float,
) -> SuiteScore:
    if (
        baseline.workloads.keys() != rerun.workloads.keys()
        or baseline.workloads.keys() != candidate.workloads.keys()
        or baseline.workloads.keys() != bounds.keys()
        or baseline.implementation_sha256 != rerun.implementation_sha256
    ):
        raise ValueError("release evidence denominators or baseline identities differ")
    scores = [
        _score_workload(
            identity,
            baseline.workloads[identity].latency_ms,
            rerun.workloads[identity].latency_ms,
            candidate.workloads[identity].latency_ms,
            candidate.workloads[identity].passed,
            bounds[identity],
            max_rerun_delta=max_rerun_delta,
        )
        for identity in sorted(bounds)
    ]
    return aggregate_suite_scores(scores)


def _score_workload(
    identity: tuple[str, str],
    baseline_latency: float | None,
    rerun_latency: float | None,
    candidate_latency: float | None,
    candidate_passed: bool,
    sol_latency: float,
    *,
    max_rerun_delta: float,
) -> WorkloadScore:
    assert baseline_latency is not None and rerun_latency is not None
    relative_delta = abs(baseline_latency - rerun_latency) / baseline_latency
    if relative_delta > max_rerun_delta:
        raise ValueError(f"independent baseline rerun drifted for {identity}")
    baseline_runtime = (baseline_latency + rerun_latency) / 2.0
    if not candidate_passed:
        return WorkloadScore(identity[0], identity[1], 0.0)
    if candidate_latency is None:
        raise ValueError(f"passing candidate has no latency for {identity}")
    score = sol_score(candidate_latency, baseline_runtime, sol_latency)
    return WorkloadScore(identity[0], identity[1], score)


def _load_model(path: Path, model: type[_Statement]) -> _Statement:
    if not path.is_file() or path.stat().st_size > MAX_SIGNED_STATEMENT_BYTES:
        raise ValueError("release statement is missing or exceeds the size limit")
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid release statement: {path.name}") from exc


__all__ = ["OfficialScoreResult", "verify_and_score_release"]
