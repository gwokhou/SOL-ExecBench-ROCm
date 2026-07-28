# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Full-corpus formal SOLAR artifact generation for a release workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sol_execbench.core.dataset.aka_contract import AKACorpusRole
from sol_execbench.core.dataset.aka_corpus import AKACorpusManifest
from sol_execbench.core.scoring.release_assembly import build_solar_index
from sol_execbench.core.scoring.release_builders import load_execution_plan
from sol_execbench.core.scoring.release_environment import (
    verify_release_source_state,
)
from sol_execbench.core.scoring.release_models import SolarIndexStatement
from sol_execbench.core.scoring.release_solar import verify_solar_index
from sol_execbench.core.solar_bridge.models import SolarWorkerRequest
from sol_execbench.core.solar_bridge.runner import run_solar_worker
from solar.graph.contracts import DEFAULT_EXTRACTION_KIND, ExtractionKind


@dataclass(frozen=True, slots=True)
class SolarReleaseResult:
    """Summary of an exact formal-manifest release build."""

    problems: int
    workloads: int
    generated: int
    resumed: int
    index_path: Path


def build_release_solar_manifests(
    workspace_root: Path,
    *,
    corpus_manifest_path: Path,
    orojenesis_home: Path,
    timeout_seconds: float = 14_400,
    resume: bool = False,
    device: str = "cuda:0",
    extraction_kind: ExtractionKind | str = DEFAULT_EXTRACTION_KIND,
) -> SolarReleaseResult:
    """Generate and index every scored workload's formal SOLAR artifacts."""
    workspace = workspace_root.resolve()
    corpus = AKACorpusManifest.load(corpus_manifest_path)
    baseline_plan = load_execution_plan(
        workspace / "baseline" / "plan.json",
        workspace_root=workspace,
    )
    verify_release_source_state(
        corpus.authored_root.parents[1],
        expected_revision=baseline_plan.source_revision,
    )
    generated = resumed = workloads = 0
    problems = 0
    for entry in corpus.entries:
        if entry.role is not AKACorpusRole.SCORED:
            continue
        problems += 1
        for workload_uuid in entry.workload_uuids:
            workloads += 1
            output = (
                workspace
                / "solar"
                / "manifests"
                / entry.relative_problem_dir
                / workload_uuid
            )
            if output.exists():
                if not resume:
                    raise FileExistsError(
                        f"SOLAR release output already exists: {output}",
                    )
                resumed += 1
                continue
            outcome = run_solar_worker(
                SolarWorkerRequest(
                    problem_dir=str(
                        (
                            corpus.authored_root / entry.relative_problem_dir
                        ).resolve(),
                    ),
                    workload_uuid=workload_uuid,
                    output_dir=str(output),
                    device=device,
                    orojenesis_home=str(orojenesis_home.resolve()),
                    extraction_kind=extraction_kind,
                ),
                timeout_seconds=timeout_seconds,
            )
            if not outcome.is_formal_publication:
                raise RuntimeError(
                    f"SOLAR failed for {entry.relative_problem_dir}/{workload_uuid}: "
                    f"{outcome.stage}/{outcome.reason_code}: {outcome.message}",
                )
            generated += 1
    index_path = workspace / "statements" / "solar.json"
    _finish_index(
        workspace,
        corpus=corpus,
        source_revision=baseline_plan.source_revision,
        index_path=index_path,
        resume=resume,
    )
    return SolarReleaseResult(
        problems=problems,
        workloads=workloads,
        generated=generated,
        resumed=resumed,
        index_path=index_path,
    )


def _finish_index(
    workspace: Path,
    *,
    corpus: AKACorpusManifest,
    source_revision: str,
    index_path: Path,
    resume: bool,
) -> None:
    if not index_path.exists():
        build_solar_index(
            workspace,
            corpus_manifest_path=corpus.path,
            source_revision=source_revision,
            output_path=index_path,
        )
        return
    if not resume:
        raise FileExistsError(
            f"SOLAR release index already exists: {index_path}",
        )
    index = SolarIndexStatement.model_validate_json(
        index_path.read_text(encoding="utf-8"),
    )
    if index.source_revision != source_revision:
        raise ValueError("resumed SOLAR release index source revision mismatch")
    verify_solar_index(index, bundle_root=workspace, corpus=corpus)


__all__ = ["SolarReleaseResult", "build_release_solar_manifests"]
