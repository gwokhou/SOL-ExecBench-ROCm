from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.bench.performance_model.evidence_manifest import (
    PerformanceRunIdentity,
)
from sol_execbench.core.bench.performance_model.lifecycle.collection_identity import (
    require_consistent_collection_gpu_identity,
)
from sol_execbench.core.platform.runtime import (
    PCIeLinkIdentity,
    PCIeTopologyIdentity,
)


def _topology(width: int = 8) -> PCIeTopologyIdentity:
    link = PCIeLinkIdentity(
        bdf="0000:03:00.0",
        current_speed_gtps=32.0,
        max_speed_gtps=32.0,
        current_width=width,
        max_width=16,
    )
    return PCIeTopologyIdentity(
        links=(link,),
        bottleneck_bdf=link.bdf,
        effective_speed_gtps=link.current_speed_gtps,
        effective_width=link.current_width,
    )


def _run(
    run_id: str,
    topology: PCIeTopologyIdentity | None,
) -> PerformanceRunIdentity:
    return PerformanceRunIdentity(
        run_id=run_id,
        definition="fixture",
        definition_sha256="a" * 64,
        workload_uuid=run_id,
        workload_sha256="b" * 64,
        solution_sha256="c" * 64,
        candidate_sha256="d" * 64,
        gpu_architecture="gfx1200",
        gpu_id="gpu-0",
        gpu_bdf="0000:03:00.0",
        pcie_topology=topology,
        rocm_version="7.2.0",
        compiler_version="HIP 7.2",
        clock_mode="locked",
        power_profile="stable_peak",
        timing_protocol="device_event_v1",
    )


def test_collection_identity_requires_one_stable_pcie_path() -> None:
    topology = _topology()

    identity = require_consistent_collection_gpu_identity(
        [_run("1" * 64, topology), _run("2" * 64, topology)]
    )

    assert identity.pcie_topology == topology


def test_collection_identity_rejects_cross_case_pcie_drift() -> None:
    with pytest.raises(ValueError, match="different GPU identities"):
        require_consistent_collection_gpu_identity(
            [_run("1" * 64, _topology(8)), _run("2" * 64, _topology(4))]
        )


def test_collection_identity_rejects_missing_pcie_path() -> None:
    with pytest.raises(ValueError, match="pcie_topology"):
        require_consistent_collection_gpu_identity([_run("1" * 64, None)])


def test_load_collection_gpu_identity_resolves_blob_backed_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``freeze`` records immutable held-out corpora as blob-backed references.

    ``load_collection_gpu_identity`` must resolve those blob-backed references
    from the lifecycle store rather than requiring path-backed evidence. This
    regression test builds a complete 220-case blob-backed corpus (one shared
    GPU identity) and asserts the loader returns that identity without raising
    ``collection hardware identity requires path evidence``.
    """
    import hashlib

    from sol_execbench.core.bench.performance_model.evidence_manifest import (
        PerformanceEvidenceArtifact,
        PerformanceEvidenceArtifactKind,
        PerformanceEvidenceManifest,
        PerformanceRunIdentity,
    )
    from sol_execbench.core.bench.performance_model.lifecycle.artifact_tree import (
        import_artifact_tree,
    )
    from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
        BlobStore,
    )
    from sol_execbench.core.bench.performance_model.lifecycle.collection_identity import (
        load_collection_gpu_identity,
    )
    from sol_execbench.core.bench.performance_model.models import (
        DiagnosticSidecarStatus,
        WorkloadKind,
    )
    from sol_execbench.core.bench.performance_model.validation_corpus import (
        BlobArtifactReference,
        DiagnosticValidationCase,
        DiagnosticValidationCorpus,
        validation_pair_id,
    )
    from sol_execbench.core.data.json_utils import atomic_write_json_value

    store_root_path = tmp_path / "store"
    monkeypatch.setenv(
        "SOL_EXECBENCH_DIAGNOSTIC_STORE",
        str(store_root_path),
    )
    store = BlobStore(store_root_path)
    topology = _topology()

    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _import_tree(
        source: Path, members: tuple[Path, ...]
    ) -> tuple[str, Path]:
        digest, _tree_manifest = import_artifact_tree(
            root=source,
            root_path=members[0],
            member_paths=members,
            store=store,
        )
        return digest, members[0]

    def _blob_ref(
        source: Path, members: tuple[Path, ...]
    ) -> BlobArtifactReference:
        digest, root = _import_tree(source, members)
        return BlobArtifactReference(
            sha256=_digest(root),
            size_bytes=root.stat().st_size,
            tree_manifest_sha256=digest,
        )

    def _evidence_ref(index: int) -> BlobArtifactReference:
        case_dir = tmp_path / "src" / f"case-{index}"
        case_dir.mkdir(parents=True)
        artifacts: list[PerformanceEvidenceArtifact] = []
        members: list[Path] = []
        for kind in (
            PerformanceEvidenceArtifactKind.TRACE,
            PerformanceEvidenceArtifactKind.TIMING,
            PerformanceEvidenceArtifactKind.ACCESS_PATTERN,
            PerformanceEvidenceArtifactKind.PROFILE_SUMMARY,
            PerformanceEvidenceArtifactKind.STATIC_EVIDENCE,
            PerformanceEvidenceArtifactKind.COUNTER_PROVENANCE,
            PerformanceEvidenceArtifactKind.COUNTER_CSV,
            PerformanceEvidenceArtifactKind.REPLAY_EVIDENCE,
        ):
            path = case_dir / f"{kind.value}.bin"
            path.write_text(f"{index}-{kind.value}", encoding="utf-8")
            members.append(path)
            artifacts.append(
                PerformanceEvidenceArtifact(
                    kind=kind,
                    path=path.name,
                    sha256=_digest(path),
                    size_bytes=path.stat().st_size,
                )
            )
        manifest = PerformanceEvidenceManifest(
            status=DiagnosticSidecarStatus.AVAILABLE,
            identity=PerformanceRunIdentity(
                run_id=f"{index:064d}",
                definition="fixture",
                definition_sha256=f"{index:064d}",
                workload_uuid=f"workload-{index}",
                workload_sha256=f"{index + 1:064d}",
                solution_sha256=f"{index + 2:064d}",
                candidate_sha256=f"{index + 3:064d}",
                gpu_architecture="gfx1200",
                gpu_id="a3ff7590-0000-1000-800f-a29c1cca1511",
                gpu_bdf="0000:03:00.0",
                pcie_topology=topology,
                rocm_version="7.2.0",
                compiler_version="HIP version: 7.2.26015-fc0010cf6a",
                clock_mode="locked",
                power_profile="stable_peak",
                timing_protocol="device_event_v1",
            ),
            artifacts=artifacts,
        )
        manifest_path = case_dir / "manifest.json"
        atomic_write_json_value(manifest_path, manifest.to_dict())
        return _blob_ref(case_dir, (manifest_path, *members))

    families = (
        WorkloadKind.ELEMENTWISE,
        WorkloadKind.TRANSPOSE,
        WorkloadKind.REDUCTION,
        WorkloadKind.MATMUL,
        WorkloadKind.SOFTMAX,
        WorkloadKind.CROSS_ENTROPY,
        WorkloadKind.INDEXED_READ,
        WorkloadKind.INDEXED_UPDATE,
        WorkloadKind.COMPOSITE,
        WorkloadKind.TRANSFORMER,
        WorkloadKind.CONCURRENT,
    )
    cases: list[DiagnosticValidationCase] = []
    index = 0
    for family in families:
        for _ in range(20):
            evidence = _evidence_ref(index)
            solar_source = tmp_path / "solar-src" / f"solar-{index}"
            solar_source.parent.mkdir(parents=True, exist_ok=True)
            solar_source.write_text(f"solar-{index}", encoding="utf-8")
            solar = _blob_ref(solar_source.parent, (solar_source,))
            cases.append(
                DiagnosticValidationCase(
                    case_id=f"case-{index}",
                    pair_id=validation_pair_id(
                        workload_sha256=f"{index + 1:064d}",
                        candidate_sha256=f"{index + 3:064d}",
                    ),
                    workload_kind=family,
                    gold_action_codes=[],
                    evidence_manifest=evidence,
                    solar_manifest=solar,
                )
            )
            index += 1

    corpus_path = tmp_path / "held_out.json"
    atomic_write_json_value(
        corpus_path,
        DiagnosticValidationCorpus(
            purpose="production",
            role="held_out",
            cases=cases,
        ).model_dump(mode="json"),
    )

    identity = load_collection_gpu_identity(
        corpus_path,
        corpus_root=tmp_path,
    )

    assert identity.gpu_id == "a3ff7590-0000-1000-800f-a29c1cca1511"
    assert identity.gpu_bdf == "0000:03:00.0"
    assert identity.pcie_topology == topology
