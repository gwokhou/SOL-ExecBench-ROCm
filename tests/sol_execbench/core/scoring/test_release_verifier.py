from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from sol_execbench.core.bench.config.benchmark_config import (
    OFFICIAL_ROCM_TIMING_PROTOCOL,
)
from sol_execbench.core.data.definition import Definition
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
from sol_execbench.core.dataset.aka_corpus import (
    AkaCorpusEntry,
    AkaCorpusManifest,
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
from sol_execbench.core.scoring.release_builders import artifact_reference
from sol_execbench.core.scoring.official_authority import official_score_availability
from sol_execbench.core.scoring.release_models import (
    ArtifactReference,
    AuthorityRole,
    BaselineStatement,
    CandidateStatement,
    ProblemRunEvidence,
    ReleaseBundle,
    RerunStatement,
    SignedStatement,
    SolarIndexStatement,
    SolarManifestEvidence,
    release_model_payload,
)
from sol_execbench.core.scoring.release_verifier import verify_and_score_release
from sol_execbench.core.solar_bridge.models import (
    FORMAL_BOUND_KIND,
    SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
)

_SOURCE_REVISION = "a" * 40
_BASELINE_ID = "rx9060xt-test-baseline"
_PROBLEM_PATH = "torch2hip/3267_doubled_matmul"


def test_caller_authored_manifest_cannot_publish_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, _, _ = _release_fixture(tmp_path)
    monkeypatch.setattr(AkaCorpusManifest, "load", lambda _path: corpus)

    report = official_score_availability(corpus.path)

    assert report["status"] == "unavailable"
    assert report["reason_code"] == "corpus_manifest_not_repository_pinned"


def test_signed_release_bundle_verifies_and_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, workspace, keys = _release_fixture(tmp_path)
    statements = _write_statements(corpus, workspace)
    signed = {
        role: _sign_statement(workspace, keys[role], path, role)
        for role, path in statements.items()
    }
    bundle = ReleaseBundle(
        corpus_manifest=artifact_reference(
            workspace, workspace / "corpus" / "manifest.yaml"
        ),
        baseline=signed[AuthorityRole.BASELINE],
        rerun=signed[AuthorityRole.RERUN],
        candidate=signed[AuthorityRole.CANDIDATE],
        solar=signed[AuthorityRole.SOLAR],
    )
    bundle_path = workspace / "release-bundle.json"
    atomic_write_json_value(bundle_path, release_model_payload(bundle))
    monkeypatch.setattr(AkaCorpusManifest, "load", lambda _path: corpus)
    _trust_fixture_manifest(monkeypatch, corpus)

    result = verify_and_score_release(
        bundle_path,
        corpus_manifest_path=corpus.path,
    )

    expected = 1.0 / (1.0 + (1.5 - 1.0) / (2.05 - 1.0))
    assert result.suite.score == pytest.approx(expected)
    assert result.suite.scored_workloads == 4
    assert result.candidate_id == "candidate-test"


def test_signed_release_rejects_reused_baseline_trace_as_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, workspace, keys = _release_fixture(tmp_path)
    shutil.copyfile(
        workspace / "baseline" / "traces" / _PROBLEM_PATH / "trace.jsonl",
        workspace / "rerun" / "traces" / _PROBLEM_PATH / "trace.jsonl",
    )
    statements = _write_statements(corpus, workspace)
    signed = {
        role: _sign_statement(workspace, keys[role], path, role)
        for role, path in statements.items()
    }
    bundle = ReleaseBundle(
        corpus_manifest=artifact_reference(
            workspace, workspace / "corpus" / "manifest.yaml"
        ),
        baseline=signed[AuthorityRole.BASELINE],
        rerun=signed[AuthorityRole.RERUN],
        candidate=signed[AuthorityRole.CANDIDATE],
        solar=signed[AuthorityRole.SOLAR],
    )
    bundle_path = workspace / "release-bundle.json"
    atomic_write_json_value(bundle_path, release_model_payload(bundle))
    monkeypatch.setattr(AkaCorpusManifest, "load", lambda _path: corpus)
    _trust_fixture_manifest(monkeypatch, corpus)

    with pytest.raises(ValueError, match="rerun reuses a baseline trace"):
        verify_and_score_release(
            bundle_path,
            corpus_manifest_path=corpus.path,
        )


def test_release_bundle_rejects_tampered_signed_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, workspace, keys = _release_fixture(tmp_path)
    statements = _write_statements(corpus, workspace)
    signed = {
        role: _sign_statement(workspace, keys[role], path, role)
        for role, path in statements.items()
    }
    baseline_path = statements[AuthorityRole.BASELINE]
    baseline_path.write_text(baseline_path.read_text() + "\n", encoding="utf-8")
    bundle = ReleaseBundle(
        corpus_manifest=artifact_reference(
            workspace, workspace / "corpus" / "manifest.yaml"
        ),
        baseline=signed[AuthorityRole.BASELINE],
        rerun=signed[AuthorityRole.RERUN],
        candidate=signed[AuthorityRole.CANDIDATE],
        solar=signed[AuthorityRole.SOLAR],
    )
    bundle_path = workspace / "release-bundle.json"
    atomic_write_json_value(bundle_path, release_model_payload(bundle))
    monkeypatch.setattr(AkaCorpusManifest, "load", lambda _path: corpus)
    _trust_fixture_manifest(monkeypatch, corpus)

    with pytest.raises(ValueError, match="size mismatch"):
        verify_and_score_release(
            bundle_path,
            corpus_manifest_path=corpus.path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hardware", "gfx1201"),
        ("rocm", "7.2.1"),
        ("triton", "3.6.1"),
    ],
)
def test_signed_release_rejects_trace_environment_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    corpus, workspace, keys = _release_fixture(tmp_path)
    trace_path = workspace / "candidate" / "traces" / _PROBLEM_PATH / "trace.jsonl"
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
                        update={"environment": environment.model_copy(update=update)}
                    )
                }
            )
        )
    atomic_write_jsonl_values(trace_path, changed)
    statements = _write_statements(corpus, workspace)
    signed = {
        role: _sign_statement(workspace, keys[role], path, role)
        for role, path in statements.items()
    }
    bundle = ReleaseBundle(
        corpus_manifest=artifact_reference(
            workspace, workspace / "corpus" / "manifest.yaml"
        ),
        baseline=signed[AuthorityRole.BASELINE],
        rerun=signed[AuthorityRole.RERUN],
        candidate=signed[AuthorityRole.CANDIDATE],
        solar=signed[AuthorityRole.SOLAR],
    )
    bundle_path = workspace / "release-bundle.json"
    atomic_write_json_value(bundle_path, release_model_payload(bundle))
    monkeypatch.setattr(AkaCorpusManifest, "load", lambda _path: corpus)
    _trust_fixture_manifest(monkeypatch, corpus)

    with pytest.raises(
        ValueError, match="release trace environment is not publication eligible"
    ):
        verify_and_score_release(
            bundle_path,
            corpus_manifest_path=corpus.path,
        )


def test_signed_candidate_failures_score_zero_without_timing_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, workspace, keys = _release_fixture(tmp_path)
    trace_path = workspace / "candidate" / "traces" / _PROBLEM_PATH / "trace.jsonl"
    traces = load_jsonl_file(Trace, trace_path)
    failed = [
        trace.model_copy(
            update={
                "evaluation": Evaluation(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    environment=trace.evaluation.environment.model_copy(
                        update={"clocks_locked": None, "timing_protocol": None}
                    ),
                    timestamp="2026-07-25T00:00:00Z",
                    log="candidate failed",
                )
            }
        )
        for trace in traces
        if trace.evaluation is not None
    ]
    atomic_write_jsonl_values(trace_path, failed)
    statements = _write_statements(corpus, workspace)
    signed = {
        role: _sign_statement(workspace, keys[role], path, role)
        for role, path in statements.items()
    }
    bundle = ReleaseBundle(
        corpus_manifest=artifact_reference(
            workspace, workspace / "corpus" / "manifest.yaml"
        ),
        baseline=signed[AuthorityRole.BASELINE],
        rerun=signed[AuthorityRole.RERUN],
        candidate=signed[AuthorityRole.CANDIDATE],
        solar=signed[AuthorityRole.SOLAR],
    )
    bundle_path = workspace / "release-bundle.json"
    atomic_write_json_value(bundle_path, release_model_payload(bundle))
    monkeypatch.setattr(AkaCorpusManifest, "load", lambda _path: corpus)
    _trust_fixture_manifest(monkeypatch, corpus)

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
        ("container_image_id", "sha256:" + "z" * 64, "immutable sha256 image ID"),
    ],
)
def test_signed_release_rejects_execution_environment_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    corpus, workspace, keys = _release_fixture(tmp_path)
    environment_path = workspace / "candidate" / "environment.json"
    environment = load_json_value(environment_path)
    environment["release_execution"][field] = value
    atomic_write_json_value(environment_path, environment)
    statements = _write_statements(corpus, workspace)
    signed = {
        role: _sign_statement(workspace, keys[role], path, role)
        for role, path in statements.items()
    }
    bundle = ReleaseBundle(
        corpus_manifest=artifact_reference(
            workspace, workspace / "corpus" / "manifest.yaml"
        ),
        baseline=signed[AuthorityRole.BASELINE],
        rerun=signed[AuthorityRole.RERUN],
        candidate=signed[AuthorityRole.CANDIDATE],
        solar=signed[AuthorityRole.SOLAR],
    )
    bundle_path = workspace / "release-bundle.json"
    atomic_write_json_value(bundle_path, release_model_payload(bundle))
    monkeypatch.setattr(AkaCorpusManifest, "load", lambda _path: corpus)
    _trust_fixture_manifest(monkeypatch, corpus)

    with pytest.raises(ValueError, match=message):
        verify_and_score_release(
            bundle_path,
            corpus_manifest_path=corpus.path,
        )


def _trust_fixture_manifest(
    monkeypatch: pytest.MonkeyPatch,
    corpus: AkaCorpusManifest,
) -> None:
    monkeypatch.setattr(
        "sol_execbench.core.scoring.release_verifier.OFFICIAL_CORPUS_MANIFEST_SHA256",
        sha256_file(corpus.path),
    )


def _release_fixture(
    tmp_path: Path,
) -> tuple[
    AkaCorpusManifest,
    Path,
    dict[AuthorityRole, tuple[Path, Path, str]],
]:
    authored = tmp_path / "authored"
    workspace = tmp_path / "workspace"
    problem = authored / _PROBLEM_PATH
    problem.mkdir(parents=True)
    workspace.mkdir()
    source = Path("problems/AMD_AKA") / _PROBLEM_PATH
    shutil.copyfile(source / "definition.json", problem / "definition.json")
    shutil.copyfile(source / "workload.jsonl", problem / "workload.jsonl")
    keys = _generate_authority_keys(authored)
    manifest_path = _write_corpus_manifest(authored, problem, keys)
    (workspace / "corpus").mkdir()
    shutil.copyfile(manifest_path, workspace / "corpus" / "manifest.yaml")
    workloads = load_jsonl_file(Workload, problem / "workload.jsonl")
    entry = AkaCorpusEntry(
        slot="3267_doubled_matmul",
        task_path="tasks/test",
        problem_name="3267_doubled_matmul",
        operation="matmul",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="gpumode",
        suite="torch2hip",
        workload_uuids=tuple(item.uuid for item in workloads),
    )
    corpus = AkaCorpusManifest(
        path=manifest_path,
        source={"revision": "b" * 40},
        execution_targets={},
        formal_analysis={"architecture_profile_sha256": "c" * 64},
        entries=(entry,),
        materialized_problem_sha256={
            _PROBLEM_PATH: {
                "definition_sha256": sha256_file(problem / "definition.json"),
                "workload_sha256": sha256_file(problem / "workload.jsonl"),
            }
        },
        formal_coverage_requirements={},
        official_scoring={"status": "available", "baseline_id": _BASELINE_ID},
    )
    _write_run_artifacts(corpus, workspace)
    _write_solar_artifacts(corpus, workspace)
    return corpus, workspace, keys


def _generate_authority_keys(
    authored: Path,
) -> dict[AuthorityRole, tuple[Path, Path, str]]:
    key_root = authored / "release-keys"
    key_root.mkdir()
    result: dict[AuthorityRole, tuple[Path, Path, str]] = {}
    for role in AuthorityRole:
        private = key_root / f"{role.value}.private.pem"
        public = key_root / f"{role.value}.public.pem"
        subprocess.run(
            [
                "openssl",
                "genpkey",
                "-algorithm",
                "ED25519",
                "-out",
                str(private),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "openssl",
                "pkey",
                "-in",
                str(private),
                "-pubout",
                "-out",
                str(public),
            ],
            check=True,
            capture_output=True,
        )
        result[role] = (private, public, sha256_file(public))
    return result


def _write_corpus_manifest(
    authored: Path,
    problem: Path,
    keys: dict[AuthorityRole, tuple[Path, Path, str]],
) -> Path:
    key_entries = []
    for role, (_, public, digest) in keys.items():
        key_entries.append(
            {
                "key_id": digest,
                "role": role.value,
                "public_key": {
                    "path": public.relative_to(authored).as_posix(),
                    "sha256": digest,
                    "size_bytes": public.stat().st_size,
                },
            }
        )
    payload = {
        "official_scoring": {"status": "available", "baseline_id": _BASELINE_ID},
        "release_authority": {
            "schema_version": 1,
            "max_rerun_relative_delta": 0.1,
            "keys": key_entries,
        },
        "definition_sha256": sha256_file(problem / "definition.json"),
    }
    path = authored / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_run_artifacts(
    corpus: AkaCorpusManifest,
    workspace: Path,
) -> None:
    problem = corpus.authored_root / _PROBLEM_PATH
    definition = Definition.model_validate_json(
        (problem / "definition.json").read_text(encoding="utf-8")
    )
    workloads = load_jsonl_file(Workload, problem / "workload.jsonl")
    baseline = _solution(definition, "baseline-solution")
    candidate = _solution(definition, "candidate-solution")
    for role, solution, latency in (
        ("baseline", baseline, 2.0),
        ("rerun", baseline, 2.1),
        ("candidate", candidate, 1.5),
    ):
        solution_path = (
            workspace / role / "implementations" / _PROBLEM_PATH / "solution.json"
        )
        trace_path = workspace / role / "traces" / _PROBLEM_PATH / "trace.jsonl"
        atomic_write_json_value(solution_path, solution.model_dump(mode="json"))
        atomic_write_jsonl_values(
            trace_path,
            [_trace(definition, solution, workload, latency) for workload in workloads],
        )
        atomic_write_json_value(
            workspace / role / "environment.json",
            _environment_evidence(),
        )


def _solution(definition: Definition, name: str) -> Solution:
    return Solution(
        name=name,
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
                    }
                }
            },
            "gpus": [
                {
                    "index": 0,
                    "gfx_target": "gfx1200",
                    "name": "AMD Radeon Graphics",
                }
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
    corpus: AkaCorpusManifest,
    workspace: Path,
) -> None:
    problem = corpus.authored_root / _PROBLEM_PATH
    definition = Definition.model_validate_json(
        (problem / "definition.json").read_text(encoding="utf-8")
    )
    for workload in load_jsonl_file(Workload, problem / "workload.jsonl"):
        root = workspace / "solar" / "manifests" / _PROBLEM_PATH / workload.uuid
        root.mkdir(parents=True)
        artifacts = []
        for name in (
            "operator_graph.yaml",
            "einsum_graph.yaml",
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
            "analysis_contract": {"require_orojenesis": True},
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


def _write_statements(
    corpus: AkaCorpusManifest,
    workspace: Path,
) -> dict[AuthorityRole, Path]:
    corpus_ref = artifact_reference(workspace, workspace / "corpus" / "manifest.yaml")
    baseline = _run_statement(
        corpus,
        workspace,
        role=AuthorityRole.BASELINE,
        corpus_ref=corpus_ref,
    )
    baseline_path = _write_payload(workspace, "baseline", baseline)
    rerun = _run_statement(
        corpus,
        workspace,
        role=AuthorityRole.RERUN,
        corpus_ref=corpus_ref,
        baseline_sha256=sha256_file(baseline_path),
    )
    candidate = _run_statement(
        corpus,
        workspace,
        role=AuthorityRole.CANDIDATE,
        corpus_ref=corpus_ref,
    )
    solar = _solar_statement(corpus, workspace, corpus_ref)
    return {
        AuthorityRole.BASELINE: baseline_path,
        AuthorityRole.RERUN: _write_payload(workspace, "rerun", rerun),
        AuthorityRole.CANDIDATE: _write_payload(workspace, "candidate", candidate),
        AuthorityRole.SOLAR: _write_payload(workspace, "solar", solar),
    }


def _run_statement(
    corpus: AkaCorpusManifest,
    workspace: Path,
    *,
    role: AuthorityRole,
    corpus_ref: ArtifactReference,
    baseline_sha256: str | None = None,
):
    role_name = role.value
    identity = corpus.materialized_problem_sha256[_PROBLEM_PATH]
    evidence = ProblemRunEvidence(
        problem_path=_PROBLEM_PATH,
        definition_sha256=identity["definition_sha256"],
        workload_sha256=identity["workload_sha256"],
        implementation=artifact_reference(
            workspace,
            workspace / role_name / "implementations" / _PROBLEM_PATH / "solution.json",
        ),
        trace=artifact_reference(
            workspace,
            workspace / role_name / "traces" / _PROBLEM_PATH / "trace.jsonl",
        ),
    )
    fields: dict[str, Any] = {
        "generated_at": "2026-07-25T00:00:00Z",
        "source_revision": _SOURCE_REVISION,
        "corpus_manifest": corpus_ref,
        "environment": artifact_reference(
            workspace, workspace / role_name / "environment.json"
        ),
        "problems": (evidence,),
    }
    if role == AuthorityRole.BASELINE:
        return BaselineStatement(**fields, baseline_id=_BASELINE_ID)
    if role == AuthorityRole.RERUN:
        assert baseline_sha256 is not None
        return RerunStatement(
            **fields,
            baseline_payload_sha256=baseline_sha256,
        )
    return CandidateStatement(**fields, candidate_id="candidate-test")


def _solar_statement(
    corpus: AkaCorpusManifest,
    workspace: Path,
    corpus_ref: ArtifactReference,
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
            )
        )
    return SolarIndexStatement(
        generated_at="2026-07-25T00:00:00Z",
        source_revision=_SOURCE_REVISION,
        corpus_manifest=corpus_ref,
        entries=tuple(entries),
    )


def _write_payload(workspace: Path, name: str, model) -> Path:
    path = workspace / "statements" / f"{name}.json"
    atomic_write_json_value(path, release_model_payload(model))
    return path


def _sign_statement(
    workspace: Path,
    key: tuple[Path, Path, str],
    payload: Path,
    role: AuthorityRole,
) -> SignedStatement:
    private, _, key_id = key
    signature = workspace / "signatures" / f"{role.value}.sig"
    signature.parent.mkdir(exist_ok=True)
    subprocess.run(
        [
            "openssl",
            "pkeyutl",
            "-sign",
            "-inkey",
            str(private),
            "-rawin",
            "-in",
            str(payload),
            "-out",
            str(signature),
        ],
        check=True,
        capture_output=True,
    )
    return SignedStatement(
        payload=artifact_reference(workspace, payload),
        signature=artifact_reference(workspace, signature),
        key_id=key_id,
        role=role,
    )
