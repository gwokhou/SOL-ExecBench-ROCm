from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from sol_execbench.core.dataset.aka_contract import AKACorpusRole
from sol_execbench.core.scoring import release_solar_runner
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarAnalysisStatus,
    SolarStage,
    SolarWorkerRequest,
)


def _corpus(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        authored_root=tmp_path / "repo" / "problems" / "AMD_AKA",
        path=tmp_path / "manifest.yaml",
        entries=(
            SimpleNamespace(
                role=AKACorpusRole.SCORED,
                relative_problem_dir=Path("p0"),
                workload_uuids=("w0", "w1"),
            ),
            SimpleNamespace(
                role=AKACorpusRole.SCORED,
                relative_problem_dir=Path("p1"),
                workload_uuids=("w2", "w3"),
            ),
        ),
    )


def _formal_outcome(request: SolarWorkerRequest) -> SolarAnalysisOutcome:
    return SolarAnalysisOutcome(
        status=SolarAnalysisStatus.ANALYZED,
        analysis_id=request.workload_uuid,
        ir_path=request.ir_path,
        output_dir=request.output_dir,
        architecture_sha256="a" * 64,
        lower_bound_seconds=0.001,
        bound_kind="capacity_constrained_tile_aware_v1",
        artifacts=tuple(
            {"path": path, "sha256": "b" * 64}
            for path in (
                "operator_graph.yaml",
                request.ir_path.graph_filename,
                "conversion-attestation.yaml",
                "solar-analysis.yaml",
            )
        ),
        publication_eligible=True,
    )


def _patch_release_inputs(
    monkeypatch: pytest.MonkeyPatch,
    corpus: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        release_solar_runner,
        "_available_physical_cpu_count",
        lambda: 16,
    )
    monkeypatch.setattr(
        release_solar_runner.AKACorpusManifest,
        "load",
        lambda _path: corpus,
    )
    monkeypatch.setattr(
        release_solar_runner,
        "load_execution_plan",
        lambda *_args, **_kwargs: SimpleNamespace(source_revision="a" * 40),
    )
    monkeypatch.setattr(
        release_solar_runner,
        "verify_release_source_state",
        lambda *_args, **_kwargs: None,
    )

    def finish(_workspace, *, index_path, **_kwargs):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text('{"generated_at":"variable"}\n')

    monkeypatch.setattr(release_solar_runner, "_finish_index", finish)


def _write_fake_publication(request: SolarWorkerRequest) -> None:
    output = Path(request.output_dir)
    output.mkdir(parents=True)
    (output / "payload.json").write_text(
        json.dumps(
            {
                "problem_dir": Path(request.problem_dir).name,
                "workload_uuid": request.workload_uuid,
                "ir_path": request.ir_path,
            },
            sort_keys=True,
        ),
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes(),
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_release_parallelism_is_opt_in_and_byte_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _corpus(tmp_path)
    _patch_release_inputs(monkeypatch, corpus)
    observed_locks: list[str | None] = []

    def worker(request, **_kwargs):
        observed_locks.append(request.device_stage_lock_path)
        _write_fake_publication(request)
        return _formal_outcome(request)

    monkeypatch.setattr(release_solar_runner, "run_solar_worker", worker)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"

    serial_result = release_solar_runner.build_release_solar_manifests(
        serial,
        corpus_manifest_path=corpus.path,
        orojenesis_home=tmp_path / "orojenesis",
    )
    serial_locks = tuple(observed_locks)
    observed_locks.clear()
    parallel_result = release_solar_runner.build_release_solar_manifests(
        parallel,
        corpus_manifest_path=corpus.path,
        orojenesis_home=tmp_path / "orojenesis",
        jobs=2,
    )

    assert serial_result.generated == parallel_result.generated == 4
    assert serial_locks == (None, None, None, None)
    assert len(set(observed_locks)) == 1
    assert observed_locks[0] is not None
    assert _tree_hashes(serial / "solar") == _tree_hashes(
        parallel / "solar",
    )
    assert (serial / "statements" / "solar.json").read_bytes() == (
        parallel / "statements" / "solar.json"
    ).read_bytes()


def test_parallel_release_runs_two_workers_and_keeps_counts_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _corpus(tmp_path)
    _patch_release_inputs(monkeypatch, corpus)
    barrier = threading.Barrier(2)
    state_lock = threading.Lock()
    active = maximum = 0

    def worker(request, **_kwargs):
        nonlocal active, maximum
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        barrier.wait(timeout=2)
        _write_fake_publication(request)
        with state_lock:
            active -= 1
        return _formal_outcome(request)

    monkeypatch.setattr(release_solar_runner, "run_solar_worker", worker)

    result = release_solar_runner.build_release_solar_manifests(
        tmp_path / "release",
        corpus_manifest_path=corpus.path,
        orojenesis_home=tmp_path / "orojenesis",
        jobs=2,
    )

    assert maximum == 2
    assert result.problems == 2
    assert result.workloads == result.generated == 4
    assert result.resumed == 0


def test_parallel_release_stops_submitting_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _corpus(tmp_path)
    _patch_release_inputs(monkeypatch, corpus)
    barrier = threading.Barrier(2)
    observed: list[str] = []

    def worker(request, **_kwargs):
        observed.append(request.workload_uuid)
        barrier.wait(timeout=2)
        if request.workload_uuid == "w0":
            return SolarAnalysisOutcome(
                status=SolarAnalysisStatus.FAILED,
                analysis_id=request.workload_uuid,
                stage=SolarStage.FORMAL_ANALYSIS,
                reason_code="formal_failed",
                message="failed safely",
            )
        _write_fake_publication(request)
        return _formal_outcome(request)

    monkeypatch.setattr(release_solar_runner, "run_solar_worker", worker)
    workspace = tmp_path / "release"

    with pytest.raises(RuntimeError, match="p0/w0.*formal_failed"):
        release_solar_runner.build_release_solar_manifests(
            workspace,
            corpus_manifest_path=corpus.path,
            orojenesis_home=tmp_path / "orojenesis",
            jobs=2,
        )

    assert set(observed) == {"w0", "w1"}
    assert not (workspace / "statements" / "solar.json").exists()


def test_parallel_release_preflights_resume_and_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _corpus(tmp_path)
    _patch_release_inputs(monkeypatch, corpus)
    workspace = tmp_path / "release"
    existing = workspace / "solar" / "manifests" / "p0" / "w0"
    existing.mkdir(parents=True)
    observed: list[str] = []

    def worker(request, **_kwargs):
        observed.append(request.workload_uuid)
        _write_fake_publication(request)
        return _formal_outcome(request)

    monkeypatch.setattr(release_solar_runner, "run_solar_worker", worker)

    with pytest.raises(FileExistsError, match="w0"):
        release_solar_runner.build_release_solar_manifests(
            workspace,
            corpus_manifest_path=corpus.path,
            orojenesis_home=tmp_path / "orojenesis",
            jobs=2,
        )
    assert observed == []

    result = release_solar_runner.build_release_solar_manifests(
        workspace,
        corpus_manifest_path=corpus.path,
        orojenesis_home=tmp_path / "orojenesis",
        jobs=2,
        resume=True,
    )

    assert result.generated == 3
    assert result.resumed == 1
    assert set(observed) == {"w1", "w2", "w3"}


def test_release_rejects_nonpositive_jobs_before_running(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="jobs must be positive"):
        release_solar_runner.build_release_solar_manifests(
            tmp_path / "release",
            corpus_manifest_path=tmp_path / "manifest.yaml",
            orojenesis_home=tmp_path / "orojenesis",
            jobs=0,
        )


def test_available_physical_cpu_count_respects_affinity(
    tmp_path: Path,
) -> None:
    topology_root = tmp_path / "cpu"
    for cpu_id, core_id in ((0, 0), (1, 0), (2, 1), (3, 1)):
        topology = topology_root / f"cpu{cpu_id}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text("0\n")
        (topology / "core_id").write_text(f"{core_id}\n")

    assert (
        release_solar_runner._available_physical_cpu_count(
            cpu_ids=frozenset({0, 1, 2, 3}),
            topology_root=topology_root,
        )
        == 2
    )
    assert (
        release_solar_runner._available_physical_cpu_count(
            cpu_ids=frozenset({1, 2}),
            topology_root=topology_root,
        )
        == 2
    )


def test_release_rejects_jobs_above_mapper_cpu_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_solar_runner,
        "_available_physical_cpu_count",
        lambda: 16,
    )

    with pytest.raises(ValueError, match="jobs 3 exceed the safe limit 2"):
        release_solar_runner.build_release_solar_manifests(
            tmp_path / "release",
            corpus_manifest_path=tmp_path / "manifest.yaml",
            orojenesis_home=tmp_path / "orojenesis",
            jobs=3,
        )


@pytest.mark.parametrize(
    ("physical_cores", "maximum_jobs", "remaining_cores"),
    (
        pytest.param(7, 1, 7, id="fewer-than-one-mapper"),
        pytest.param(8, 1, 0, id="one-exact-mapper"),
        pytest.param(15, 1, 7, id="one-mapper-with-remainder"),
        pytest.param(16, 2, 0, id="two-exact-mappers"),
        pytest.param(20, 2, 4, id="two-mappers-with-remainder"),
        pytest.param(24, 3, 0, id="three-exact-mappers"),
    ),
)
def test_safe_release_jobs_limit_uses_only_complete_mapper_slots(
    physical_cores: int,
    maximum_jobs: int,
    remaining_cores: int,
) -> None:
    assert release_solar_runner._safe_release_jobs_limit(physical_cores) == (
        maximum_jobs,
        remaining_cores,
    )


def test_safe_release_jobs_limit_rejects_nonpositive_core_count() -> None:
    with pytest.raises(ValueError, match="physical CPU cores must be positive"):
        release_solar_runner._safe_release_jobs_limit(0)


def test_safe_release_jobs_limit_rejects_nonpositive_mapper_threads() -> None:
    with pytest.raises(ValueError, match="mapper threads must be positive"):
        release_solar_runner._safe_release_jobs_limit(
            16,
            mapper_threads=0,
        )


def test_jobs_limit_reports_nondivisible_core_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_solar_runner,
        "_available_physical_cpu_count",
        lambda: 20,
    )

    with pytest.raises(
        ValueError,
        match="safe limit 2.*leaving 4 physical cores",
    ):
        release_solar_runner._validate_release_jobs(3)


def test_parallel_release_fails_closed_without_cpu_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_solar_runner,
        "_available_physical_cpu_count",
        lambda: None,
    )

    with pytest.raises(ValueError, match="physical CPU cores could not"):
        release_solar_runner.build_release_solar_manifests(
            tmp_path / "release",
            corpus_manifest_path=tmp_path / "manifest.yaml",
            orojenesis_home=tmp_path / "orojenesis",
            jobs=2,
        )
