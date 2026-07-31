from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from sol_execbench.core.bench.config.benchmark_config import (
    OFFICIAL_ROCM_TIMING_PROTOCOL,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    atomic_write_jsonl_values,
    load_json_value,
    load_jsonl_file,
)
from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.data.solution_models import (
    BuildSpec,
    SourceFile,
    SupportedHardware,
    SupportedLanguages,
)
from sol_execbench.core.data.trace import (
    CacheClearEvidence,
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_contract import (
    AKAFusionDepth,
    AKAOperation,
    AKAPassKind,
    AKAReleasePolicy,
    AKASourceFamily,
    AKASuite,
)
from sol_execbench.core.dataset.aka_corpus import (
    AKACorpusEntry,
    AKACorpusManifest,
)
from sol_execbench.core.integrity import sha256_bytes, sha256_file
from sol_execbench.core.integrity.schema_versions import (
    RELEASE_ENVIRONMENT_SCHEMA_VERSION,
)
from sol_execbench.core.platform.rdna4_validation import (
    RDNA4_VALIDATION_GFX_TARGET,
    RDNA4_VALIDATION_HIP_VERSION,
    RDNA4_VALIDATION_PCI_DEVICE_ID,
    RDNA4_VALIDATION_PCI_VENDOR_ID,
    RDNA4_VALIDATION_ROCM_VERSION,
    RDNA4_VALIDATION_TORCH_VERSION,
    RDNA4_VALIDATION_TRITON_VERSION,
)
from sol_execbench.core.scoring.official_scoring import (
    official_score_availability,
)
from sol_execbench.core.scoring.release_assembly import assemble_release_bundle
from sol_execbench.core.scoring.release_builders import (
    artifact_reference,
    reference_baseline_solution,
)
from sol_execbench.core.scoring.release_models import (
    ArtifactReference,
    BaselineStatement,
    CandidateStatement,
    ProblemRunEvidence,
    ReleaseArtifactKind,
    ReleaseBundle,
    SolarIndexStatement,
    SolarManifestEvidence,
    release_model_payload,
)
from sol_execbench.core.scoring.release_solar import verify_solar_index
from sol_execbench.core.scoring.release_verifier import verify_and_score_release
from sol_execbench.core.solar_bridge.models import (
    FORMAL_BOUND_KIND,
    SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
    IRPath,
)

_SOURCE_REVISION = "a" * 40
_BASELINE_ID = "rx9060xt-test-baseline"
_PROBLEM_PATH = "torch2hip/3267_doubled_matmul"


def test_caller_authored_manifest_cannot_publish_scoring_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, _ = _release_fixture(tmp_path)
    monkeypatch.setattr(AKACorpusManifest, "load", lambda _path: corpus)

    report = official_score_availability(corpus.path)

    assert report["policy"]["authorized"] is False
    assert (
        report["policy"]["reason_code"]
        == "corpus_manifest_not_repository_pinned"
    )
    assert report["verifier"]["requires_signatures"] is False


def test_content_addressed_release_bundle_verifies_and_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, workspace = _release_fixture(tmp_path)
    bundle_path = _write_bundle(corpus, workspace)
    _trust_fixture(monkeypatch, corpus)

    result = verify_and_score_release(
        bundle_path,
        corpus_manifest_path=corpus.path,
    )

    expected = 1.0 / (1.0 + (1.5 - 1.0) / (2.0 - 1.0))
    assert result.suite.score == pytest.approx(expected)
    assert result.suite.scored_workloads == 4
    assert result.candidate_id == "candidate-test"


def test_make_fx_aten_release_index_round_trips_and_verifies(
    tmp_path: Path,
) -> None:
    corpus, workspace = _release_fixture(tmp_path)
    _write_solar_artifacts(corpus, workspace, IRPath.MAKE_FX_ATEN)
    index = _solar_statement(
        corpus,
        workspace,
        artifact_reference(
            workspace,
            workspace / "corpus" / "manifest.yaml",
        ),
        IRPath.MAKE_FX_ATEN,
    )
    round_tripped = SolarIndexStatement.model_validate(
        release_model_payload(index),
    )

    assert round_tripped.ir_path is IRPath.MAKE_FX_ATEN
    assert verify_solar_index(
        round_tripped,
        bundle_root=workspace,
        corpus=corpus,
    )


def test_release_index_accepts_content_addressed_orojenesis_evidence(
    tmp_path: Path,
) -> None:
    corpus, workspace = _release_fixture(tmp_path)
    _write_solar_artifacts(corpus, workspace)
    for workload_uuid in corpus.entries[0].workload_uuids:
        root = workspace / "solar" / "manifests" / _PROBLEM_PATH / workload_uuid
        evidence = root / "orojenesis" / "mm" / "problem.yaml"
        evidence.parent.mkdir(parents=True)
        evidence.write_text("problem: mm\n", encoding="utf-8")
        manifest = yaml.safe_load((root / "manifest.yaml").read_text())
        manifest["artifacts"].append(
            {
                "path": "orojenesis/mm/problem.yaml",
                "sha256": sha256_file(evidence),
            },
        )
        (root / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
    index = _solar_statement(
        corpus,
        workspace,
        artifact_reference(
            workspace,
            workspace / "corpus" / "manifest.yaml",
        ),
    )

    assert verify_solar_index(
        index,
        bundle_root=workspace,
        corpus=corpus,
    )


def test_publisher_assembly_verifies_and_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, workspace = _release_fixture(tmp_path)
    statements = _write_statements(corpus, workspace)
    _trust_fixture(monkeypatch, corpus)

    bundle_path = assemble_release_bundle(
        workspace,
        corpus_manifest_path=corpus.path,
        statement_paths=statements,
        output_path=workspace / "release-bundle.json",
    )
    result = verify_and_score_release(
        bundle_path,
        corpus_manifest_path=corpus.path,
    )

    assert result.baseline_id == _BASELINE_ID
    assert result.suite.scored_workloads == 4


def test_release_bundle_rejects_tampered_statement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, workspace = _release_fixture(tmp_path)
    bundle_path = _write_bundle(corpus, workspace)
    baseline_path = workspace / "statements" / "baseline.json"
    baseline_path.write_text(baseline_path.read_text() + "\n", encoding="utf-8")
    _trust_fixture(monkeypatch, corpus)

    with pytest.raises(ValueError, match="size mismatch"):
        verify_and_score_release(bundle_path, corpus_manifest_path=corpus.path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hardware", "gfx1201"),
        ("rocm", "7.2.1"),
        ("triton", "3.6.1"),
    ],
)
def test_release_rejects_trace_environment_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    corpus, workspace = _release_fixture(tmp_path)
    trace_path = (
        workspace / "candidate" / "traces" / _PROBLEM_PATH / "trace.jsonl"
    )
    traces = load_jsonl_file(Trace, trace_path)
    changed = []
    for trace in traces:
        assert trace.evaluation is not None
        environment = trace.evaluation.environment
        update: dict[str, object] = (
            {"hardware": value}
            if field == "hardware"
            else {"libs": {**environment.libs, field: value}}
        )
        changed.append(
            trace.model_copy(
                update={
                    "evaluation": trace.evaluation.model_copy(
                        update={
                            "environment": environment.model_copy(
                                update=update
                            ),
                        },
                    ),
                },
            ),
        )
    atomic_write_jsonl_values(trace_path, changed)
    bundle_path = _write_bundle(corpus, workspace)
    _trust_fixture(monkeypatch, corpus)

    with pytest.raises(
        ValueError,
        match="release trace environment is not publication eligible",
    ):
        verify_and_score_release(bundle_path, corpus_manifest_path=corpus.path)


def test_candidate_failures_score_zero_without_timing_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, workspace = _release_fixture(tmp_path)
    trace_path = (
        workspace / "candidate" / "traces" / _PROBLEM_PATH / "trace.jsonl"
    )
    traces = load_jsonl_file(Trace, trace_path)
    failed = [
        trace.model_copy(
            update={
                "evaluation": Evaluation(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    environment=trace.evaluation.environment.model_copy(
                        update={"clocks_locked": None, "timing_protocol": None},
                    ),
                    timestamp="2026-07-25T00:00:00Z",
                    log="candidate failed",
                ),
            },
        )
        for trace in traces
        if trace.evaluation is not None
    ]
    atomic_write_jsonl_values(trace_path, failed)
    bundle_path = _write_bundle(corpus, workspace)
    _trust_fixture(monkeypatch, corpus)

    result = verify_and_score_release(
        bundle_path,
        corpus_manifest_path=corpus.path,
    )

    assert result.suite.score == 0.0
    assert result.suite.scored_workloads == 4


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_revision", "b" * 40, "source revision mismatch"),
        ("source_tree_clean", False, "environment contract mismatch"),
        (
            "container_image_id",
            "sha256:" + "e" * 64,
            "different environment identities",
        ),
        (
            "container_image_id",
            "sha256:" + "z" * 64,
            "immutable sha256 image ID",
        ),
    ],
)
def test_release_rejects_execution_environment_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    corpus, workspace = _release_fixture(tmp_path)
    environment_path = workspace / "candidate" / "environment.json"
    environment = load_json_value(environment_path)
    environment["release_execution"][field] = value
    atomic_write_json_value(environment_path, environment)
    bundle_path = _write_bundle(corpus, workspace)
    _trust_fixture(monkeypatch, corpus)

    with pytest.raises(ValueError, match=message):
        verify_and_score_release(bundle_path, corpus_manifest_path=corpus.path)


def test_release_rejects_noncanonical_baseline_solution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, workspace = _release_fixture(tmp_path)
    path = (
        workspace
        / "baseline"
        / "implementations"
        / _PROBLEM_PATH
        / "solution.json"
    )
    solution = Solution.model_validate_json(path.read_text(encoding="utf-8"))
    atomic_write_json_value(
        path,
        solution.model_copy(
            update={"author": "not the release baseline"},
        ).model_dump(
            mode="json",
        ),
    )
    bundle_path = _write_bundle(corpus, workspace)
    _trust_fixture(monkeypatch, corpus)

    with pytest.raises(
        ValueError,
        match="baseline is not the canonical reference",
    ):
        verify_and_score_release(bundle_path, corpus_manifest_path=corpus.path)


def test_release_rejects_wrong_baseline_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, workspace = _release_fixture(tmp_path)
    statements = _write_statements(corpus, workspace, baseline_id="wrong")
    bundle_path = _write_bundle_from_statements(workspace, statements)
    _trust_fixture(monkeypatch, corpus)

    with pytest.raises(ValueError, match="baseline_id is not corpus-pinned"):
        verify_and_score_release(bundle_path, corpus_manifest_path=corpus.path)


def test_release_rejects_source_revision_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, workspace = _release_fixture(tmp_path)
    statements = _write_statements(
        corpus,
        workspace,
        candidate_revision="b" * 40,
    )
    bundle_path = _write_bundle_from_statements(workspace, statements)
    _trust_fixture(monkeypatch, corpus)

    with pytest.raises(ValueError, match="source revisions do not match"):
        verify_and_score_release(bundle_path, corpus_manifest_path=corpus.path)


def test_release_bundle_schema_rejects_legacy_signature_fields(
    tmp_path: Path,
) -> None:
    corpus, workspace = _release_fixture(tmp_path)
    statements = _write_statements(corpus, workspace)
    baseline = artifact_reference(
        workspace,
        statements[ReleaseArtifactKind.BASELINE],
    ).model_dump(mode="json")
    baseline["signature"] = {"path": "baseline.sig", "sha256": "0" * 64}

    with pytest.raises(ValidationError):
        ReleaseBundle.model_validate(
            {
                "corpus_manifest": artifact_reference(
                    workspace,
                    workspace / "corpus" / "manifest.yaml",
                ),
                "baseline": baseline,
                "candidate": artifact_reference(
                    workspace,
                    statements[ReleaseArtifactKind.CANDIDATE],
                ),
                "solar": artifact_reference(
                    workspace,
                    statements[ReleaseArtifactKind.SOLAR],
                ),
            },
        )


def _trust_fixture(
    monkeypatch: pytest.MonkeyPatch,
    corpus: AKACorpusManifest,
) -> None:
    monkeypatch.setattr(AKACorpusManifest, "load", lambda _path: corpus)
    monkeypatch.setattr(
        "sol_execbench.core.scoring.release_verifier.OFFICIAL_CORPUS_MANIFEST_SHA256",
        sha256_file(corpus.path),
    )


def _release_fixture(tmp_path: Path) -> tuple[AKACorpusManifest, Path]:
    authored = tmp_path / "authored"
    workspace = tmp_path / "workspace"
    problem = authored / _PROBLEM_PATH
    problem.mkdir(parents=True)
    workspace.mkdir()
    source = Path("problems/AMD_AKA") / _PROBLEM_PATH
    shutil.copyfile(source / "definition.json", problem / "definition.json")
    shutil.copyfile(source / "workload.jsonl", problem / "workload.jsonl")
    manifest_path = _write_corpus_manifest(authored, problem)
    (workspace / "corpus").mkdir()
    shutil.copyfile(manifest_path, workspace / "corpus" / "manifest.yaml")
    workloads = load_jsonl_file(Workload, problem / "workload.jsonl")
    entry = AKACorpusEntry(
        slot="3267_doubled_matmul",
        task_path="tasks/test",
        problem_name="3267_doubled_matmul",
        operation=AKAOperation.MATMUL,
        input_dtypes=(DType.FLOAT32,),
        output_dtypes=(DType.FLOAT32,),
        capabilities=(),
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.GPUMODE,
        suite=AKASuite.TORCH2HIP,
        workload_uuids=tuple(item.uuid for item in workloads),
    )
    corpus = AKACorpusManifest(
        path=manifest_path,
        source={"revision": "b" * 40},
        execution_targets={},
        formal_analysis={"architecture_profile_sha256": "c" * 64},
        entries=(entry,),
        materialized_problem_sha256={
            _PROBLEM_PATH: {
                "definition_sha256": sha256_file(problem / "definition.json"),
                "workload_sha256": sha256_file(problem / "workload.jsonl"),
            },
        },
        formal_coverage_requirements={},
        official_scoring={
            "status": "available",
            "release_policy": str(
                AKAReleasePolicy.CONTENT_ADDRESSED_PUBLISHER_V1,
            ),
            "baseline_id": _BASELINE_ID,
        },
    )
    _write_run_artifacts(corpus, workspace)
    _write_solar_artifacts(corpus, workspace)
    return corpus, workspace


def _write_corpus_manifest(authored: Path, problem: Path) -> Path:
    payload = {
        "official_scoring": {
            "status": "available",
            "release_policy": str(
                AKAReleasePolicy.CONTENT_ADDRESSED_PUBLISHER_V1,
            ),
            "baseline_id": _BASELINE_ID,
        },
        "definition_sha256": sha256_file(problem / "definition.json"),
    }
    path = authored / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_run_artifacts(corpus: AKACorpusManifest, workspace: Path) -> None:
    problem = corpus.authored_root / _PROBLEM_PATH
    definition = Definition.model_validate_json(
        (problem / "definition.json").read_text(encoding="utf-8"),
    )
    workloads = load_jsonl_file(Workload, problem / "workload.jsonl")
    baseline = reference_baseline_solution(definition)
    candidate = _candidate_solution(definition)
    for kind, solution, latency in (
        ("baseline", baseline, 2.0),
        ("candidate", candidate, 1.5),
    ):
        solution_path = (
            workspace
            / kind
            / "implementations"
            / _PROBLEM_PATH
            / "solution.json"
        )
        trace_path = workspace / kind / "traces" / _PROBLEM_PATH / "trace.jsonl"
        atomic_write_json_value(solution_path, solution.model_dump(mode="json"))
        atomic_write_jsonl_values(
            trace_path,
            [
                _trace(definition, solution, workload, latency)
                for workload in workloads
            ],
        )
        atomic_write_json_value(
            workspace / kind / "environment.json",
            _environment_evidence(),
        )


def _candidate_solution(definition: Definition) -> Solution:
    return Solution(
        name="candidate-solution",
        definition=definition.name,
        author="release-test",
        spec=BuildSpec(
            languages=[SupportedLanguages.PYTORCH],
            target_hardware=[SupportedHardware.GFX1200],
            entry_point="kernel.py::run",
            dependencies=["torch"],
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="kernel.py", content=definition.reference)],
    )


def _trace(
    definition: Definition,
    solution: Solution,
    workload: Workload,
    latency: float,
) -> Trace:
    environment = Environment(
        hardware=RDNA4_VALIDATION_GFX_TARGET,
        libs={
            "torch": RDNA4_VALIDATION_TORCH_VERSION,
            "hip": RDNA4_VALIDATION_HIP_VERSION,
            "rocm": RDNA4_VALIDATION_ROCM_VERSION,
            "triton": RDNA4_VALIDATION_TRITON_VERSION,
        },
        execution_isolation="container",
        clocks_locked=True,
        timing_protocol=OFFICIAL_ROCM_TIMING_PROTOCOL,
    )
    performance = Performance(
        latency_ms=latency,
        reference_latency_ms=3.0,
        warmup_runs=10,
        timed_iterations=50,
        timed_iterations_per_trial=[50, 50, 50],
        trials=3,
        statistic="mean",
        timed_outputs_validated=True,
        cache_clear=CacheClearEvidence(
            detected_l2_bytes=1024,
            clear_buffer_bytes=2048,
            source="torch_device_properties",
        ),
    )
    return Trace(
        definition=definition.name,
        workload=workload,
        solution=solution.name,
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            environment=environment,
            timestamp="2026-07-25T00:00:00Z",
            correctness=Correctness(),
            performance=performance,
        ),
    )


def _environment_evidence() -> dict[str, object]:
    return {
        "status": "available",
        "release_execution": {
            "schema_version": RELEASE_ENVIRONMENT_SCHEMA_VERSION,
            "source_revision": _SOURCE_REVISION,
            "source_tree_clean": True,
            "container_image_id": "sha256:" + "d" * 64,
        },
        "snapshot": {
            "tools": {
                "amd-smi": {
                    "parsed": {
                        "pci_vendor_ids": [RDNA4_VALIDATION_PCI_VENDOR_ID],
                        "pci_device_ids": [RDNA4_VALIDATION_PCI_DEVICE_ID],
                    },
                },
            },
            "gpus": [
                {
                    "index": 0,
                    "gfx_target": "gfx1200",
                    "name": "AMD Radeon Graphics",
                },
            ],
            "pytorch": {
                "available": True,
                "device_count": 1,
                "device_name": "AMD Radeon Graphics",
                "gfx_target": "gfx1200",
                "torch_version": RDNA4_VALIDATION_TORCH_VERSION,
                "hip_version": RDNA4_VALIDATION_HIP_VERSION,
            },
            "rocm": {"version": RDNA4_VALIDATION_ROCM_VERSION},
        },
    }


def _write_solar_artifacts(
    corpus: AKACorpusManifest,
    workspace: Path,
    ir_path: IRPath = IRPath.TORCHVIEW_EXTENDED_EINSUM,
) -> None:
    problem = corpus.authored_root / _PROBLEM_PATH
    definition = Definition.model_validate_json(
        (problem / "definition.json").read_text(encoding="utf-8"),
    )
    for workload in load_jsonl_file(Workload, problem / "workload.jsonl"):
        root = workspace / "solar" / "manifests" / _PROBLEM_PATH / workload.uuid
        root.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for name in (
            "operator_graph.yaml",
            ir_path.graph_filename,
            "conversion-attestation.yaml",
            "solar-analysis.yaml",
        ):
            path = root / name
            path.write_text(f"artifact: {name}\n", encoding="utf-8")
            artifacts.append({"path": name, "sha256": sha256_file(path)})
        manifest = {
            "schema_version": SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
            "analysis_id": f"{definition.name}:{workload.uuid}",
            "architecture_sha256": "c" * 64,
            "reference": {
                "name": "test",
                "sha256": sha256_bytes(definition.reference.encode()),
            },
            "analysis_contract": {
                "ir_path": ir_path.value,
                "extraction_kind": ir_path.extraction_kind.value,
                "precision": "fp32",
                "ir_kind": ir_path.ir_kind.value,
                "trace_seed": 200,
                "verification_seeds": [11, 29, 47],
                "atol": 1e-2,
                "rtol": 1e-2,
                "required_matched_ratio": 0.99,
                "max_error_cap": None,
                "allow_negative_inf": False,
                "preserved_input_indices": [],
                "require_orojenesis": True,
            },
            "sol_score_eligible": True,
            "publication_eligible": True,
            "artifacts": artifacts,
            "bound": {
                "seconds": 0.001,
                "kind": FORMAL_BOUND_KIND,
                "limiting_resource": "memory",
            },
        }
        (root / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )


def _write_bundle(corpus: AKACorpusManifest, workspace: Path) -> Path:
    return _write_bundle_from_statements(
        workspace,
        _write_statements(corpus, workspace),
    )


def _write_bundle_from_statements(
    workspace: Path,
    statements: dict[ReleaseArtifactKind, Path],
) -> Path:
    bundle = ReleaseBundle(
        corpus_manifest=artifact_reference(
            workspace,
            workspace / "corpus" / "manifest.yaml",
        ),
        baseline=artifact_reference(
            workspace,
            statements[ReleaseArtifactKind.BASELINE],
        ),
        candidate=artifact_reference(
            workspace,
            statements[ReleaseArtifactKind.CANDIDATE],
        ),
        solar=artifact_reference(
            workspace,
            statements[ReleaseArtifactKind.SOLAR],
        ),
    )
    bundle_path = workspace / "release-bundle.json"
    atomic_write_json_value(bundle_path, release_model_payload(bundle))
    return bundle_path


def _write_statements(
    corpus: AKACorpusManifest,
    workspace: Path,
    *,
    baseline_id: str = _BASELINE_ID,
    candidate_revision: str = _SOURCE_REVISION,
) -> dict[ReleaseArtifactKind, Path]:
    corpus_ref = artifact_reference(
        workspace,
        workspace / "corpus" / "manifest.yaml",
    )
    baseline = _run_statement(
        corpus,
        workspace,
        kind=ReleaseArtifactKind.BASELINE,
        corpus_ref=corpus_ref,
        baseline_id=baseline_id,
    )
    candidate = _run_statement(
        corpus,
        workspace,
        kind=ReleaseArtifactKind.CANDIDATE,
        corpus_ref=corpus_ref,
        source_revision=candidate_revision,
    )
    solar = _solar_statement(corpus, workspace, corpus_ref)
    return {
        ReleaseArtifactKind.BASELINE: _write_payload(
            workspace,
            ReleaseArtifactKind.BASELINE,
            baseline,
        ),
        ReleaseArtifactKind.CANDIDATE: _write_payload(
            workspace,
            ReleaseArtifactKind.CANDIDATE,
            candidate,
        ),
        ReleaseArtifactKind.SOLAR: _write_payload(
            workspace,
            ReleaseArtifactKind.SOLAR,
            solar,
        ),
    }


def _run_statement(
    corpus: AKACorpusManifest,
    workspace: Path,
    *,
    kind: ReleaseArtifactKind,
    corpus_ref: ArtifactReference,
    baseline_id: str = _BASELINE_ID,
    source_revision: str = _SOURCE_REVISION,
) -> BaselineStatement | CandidateStatement:
    identity = corpus.materialized_problem_sha256[_PROBLEM_PATH]
    evidence = ProblemRunEvidence(
        problem_path=_PROBLEM_PATH,
        definition_sha256=identity["definition_sha256"],
        workload_sha256=identity["workload_sha256"],
        implementation=artifact_reference(
            workspace,
            workspace
            / kind
            / "implementations"
            / _PROBLEM_PATH
            / "solution.json",
        ),
        trace=artifact_reference(
            workspace,
            workspace / kind / "traces" / _PROBLEM_PATH / "trace.jsonl",
        ),
    )
    fields: dict[str, Any] = {
        "generated_at": "2026-07-25T00:00:00Z",
        "source_revision": source_revision,
        "corpus_manifest": corpus_ref,
        "environment": artifact_reference(
            workspace,
            workspace / kind / "environment.json",
        ),
        "problems": (evidence,),
    }
    if kind is ReleaseArtifactKind.BASELINE:
        return BaselineStatement(**fields, baseline_id=baseline_id)
    return CandidateStatement(**fields, candidate_id="candidate-test")


def _solar_statement(
    corpus: AKACorpusManifest,
    workspace: Path,
    corpus_ref: ArtifactReference,
    ir_path: IRPath = IRPath.TORCHVIEW_EXTENDED_EINSUM,
) -> SolarIndexStatement:
    entries = []
    for workload_uuid in corpus.entries[0].workload_uuids:
        manifest = (
            workspace
            / "solar"
            / "manifests"
            / _PROBLEM_PATH
            / workload_uuid
            / "manifest.yaml"
        )
        entries.append(
            SolarManifestEvidence(
                problem_path=_PROBLEM_PATH,
                workload_uuid=workload_uuid,
                manifest=artifact_reference(workspace, manifest),
            ),
        )
    return SolarIndexStatement(
        generated_at="2026-07-25T00:00:00Z",
        source_revision=_SOURCE_REVISION,
        ir_path=ir_path,
        corpus_manifest=corpus_ref,
        entries=tuple(entries),
    )


def _write_payload(
    workspace: Path,
    kind: ReleaseArtifactKind,
    model: BaselineStatement | CandidateStatement | SolarIndexStatement,
) -> Path:
    path = workspace / "statements" / f"{kind}.json"
    atomic_write_json_value(path, release_model_payload(model))
    return path
