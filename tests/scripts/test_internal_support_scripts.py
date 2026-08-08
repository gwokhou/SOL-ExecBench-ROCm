from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sol_execbench.core.bench.performance_model.corpus_preflight import (
    preflight,
)
from sol_execbench.core.integrity.schema_versions import SchemaVersion


def test_aka_author_seed_helpers_and_coverage_inventory(load_script) -> None:
    author = load_script("scripts/internal/aka_author_seed.py")

    assert author._ax_var("rows") == {"type": "var", "description": "rows"}
    assert author._ax_const(4) == {
        "type": "const",
        "value": 4,
        "description": "",
    }
    assert author._ax_expr("M * 2", "double") == {
        "type": "expr",
        "expression": "M * 2",
        "description": "double",
    }
    assert author._wl({"M": 4}, {"x": "random"}) == {
        "axes": {"M": 4},
        "inputs": {"x": "random"},
    }

    coverage = author._coverage_axes(author.SPECS[:3])
    assert set(coverage) == {
        "operation",
        "input_dtype",
        "output_dtype",
        "capability",
        "pass_kind",
        "fusion_depth",
        "source_family",
        "suite",
    }
    for axis in (
        "operation",
        "pass_kind",
        "fusion_depth",
        "source_family",
        "suite",
    ):
        assert sum(coverage[axis].values()) == 3
    assert sum(coverage["input_dtype"].values()) >= 3
    assert sum(coverage["output_dtype"].values()) >= 3


def test_aka_calibration_variation_handles_values_and_empty_outputs(
    load_script,
) -> None:
    calibration = load_script("scripts/internal/aka_calibrate_tolerances.py")
    anchor = (torch.tensor([1.0, 2.0]), torch.tensor([]))
    observed = (torch.tensor([1.5, 1.0]), torch.tensor([]))

    metrics = calibration._variation(
        anchor,
        observed,
        ["float32", "float32"],
    )

    assert metrics[0][0] == pytest.approx(1.0)
    assert metrics[0][1] == pytest.approx(0.5)
    assert metrics[1] == (0.0, 0.0)


def test_orojenesis_provenance_compiler_identity(
    load_script,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = load_script("scripts/internal/orojenesis/write_provenance.py")
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="clang version 19.0\nCopyright\n",
        ),
    )

    assert provenance._compiler_identity(tmp_path / "compiler") == (
        "clang version 19.0"
    )


def test_matrix_schema_export_writes_single_and_complete_sets(
    load_script,
    tmp_path: Path,
) -> None:
    exporter = load_script("scripts/internal/reports/export_matrix_schema.py")
    entry_path = tmp_path / "entry.json"
    all_path = tmp_path / "schemas"

    assert (
        exporter.main(["--model", "matrix-entry", "--output", str(entry_path)])
        == 0
    )
    assert exporter.main(["--model", "all", "--output-dir", str(all_path)]) == 0

    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    assert entry["type"] == "object"
    assert {
        "matrix-entry.schema.json",
        "rocm-compatibility-matrix-report.schema.json",
    } == {path.name for path in all_path.iterdir()}


@pytest.mark.parametrize(
    "arguments",
    (
        ["--model", "all", "--output", "schema.json"],
        ["--model", "all"],
        ["--model", "report", "--output-dir", "schemas"],
        ["--model", "report"],
    ),
)
def test_matrix_schema_export_rejects_incompatible_destinations(
    load_script,
    arguments: list[str],
) -> None:
    exporter = load_script("scripts/internal/reports/export_matrix_schema.py")

    with pytest.raises(SystemExit) as exc_info:
        exporter.main(arguments)

    assert exc_info.value.code == 2


def test_clock_lock_workload_amd_smi_and_log_are_bounded(
    load_script,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workload = load_script(
        "scripts/internal/rdna4/rdna4_clock_lock_workload_test.py",
    )
    seen: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return SimpleNamespace(stdout="clock: 2900 MHz\n")

    monkeypatch.setattr(workload, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        workload.shutil,
        "which",
        lambda _name: "/usr/bin/amd-smi",
    )
    monkeypatch.setattr(workload.subprocess, "run", fake_run)

    assert workload.amd_smi("metric", "-c") == "clock: 2900 MHz\n"
    workload.log("clock/state", "clock: 2900 MHz\n")

    assert seen["command"] == ["/usr/bin/amd-smi", "metric", "-c"]
    assert seen["kwargs"]["timeout"] == 30
    assert (tmp_path / "clock_state.txt").read_text() == "clock: 2900 MHz\n"
    assert "[clock/state]" in capsys.readouterr().out


def test_rdna4_diagnostic_corpus_design_is_preregistered_and_stratified(
    load_script,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )
    universe_start = 160
    development = corpus._cases("development", universe_start)
    held_out = corpus._cases("held_out", universe_start)
    all_cases = [*development, *held_out]

    assert len(development) == 440
    assert len(held_out) == 220
    for family in corpus.FAMILIES:
        for phase in ("point_fit", "conformal", "held_out"):
            selected = [
                case
                for case in all_cases
                if case.family is family and case.phase == phase
            ]
            assert len(selected) == 20

    for start in range(universe_start, universe_start + 60, 3):
        assert {
            corpus._phase(index, universe_start)
            for index in range(start, start + 3)
        } == {
            "point_fit",
            "conformal",
            "held_out",
        }

    reduction = [
        case
        for case in all_cases
        if case.family is corpus.WorkloadKind.REDUCTION
    ]
    expected_widths = {32, 64, 128, 256, 512, 1024}
    for phase in ("point_fit", "conformal", "held_out"):
        widths = [case.axes["N"] for case in reduction if case.phase == phase]
        assert set(widths) == expected_widths
        assert max(widths.count(width) for width in expected_widths) == 4
        assert min(widths.count(width) for width in expected_widths) == 3

    design = corpus._design_payload(universe_start)
    assert design["configuration_frozen_before_collection"] is True
    assert len(design["cases"]) == 660

    previous = {
        case.workload_uuid
        for case in [
            *corpus._cases("development", 100),
            *corpus._cases("held_out", 100),
        ]
    }
    assert previous.isdisjoint(case.workload_uuid for case in all_cases)


def test_rdna4_diagnostic_preregistration_is_immutable(
    load_script,
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )
    monkeypatch.setenv(
        "SOL_EXECBENCH_DIAGNOSTIC_STORE", str(tmp_path / "store")
    )
    corpus._preregister(tmp_path, 160)
    design_path = tmp_path / "design.json"

    corpus._preregister(tmp_path, 160)
    with pytest.raises(ValueError, match="differs"):
        corpus._preregister(tmp_path, 220)
    design_path.write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="differs"):
        corpus._preregister(tmp_path, 160)


def test_rdna4_diagnostic_packaged_templates_prepare_full_corpus(
    load_script,
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )
    monkeypatch.setenv(
        "SOL_EXECBENCH_DIAGNOSTIC_STORE", str(tmp_path / "store")
    )
    corpus._preregister(tmp_path, 160)

    corpus._prepare(tmp_path)

    summary = preflight(tmp_path)
    assert summary.schema_version == SchemaVersion.DIAGNOSTIC_CORPUS_PREFLIGHT
    assert summary.cases == 660
    assert summary.families == 11


def _write_promotion_source(corpus, root: Path, role: str, offset: int) -> Path:
    cases = []
    for family_index, family in enumerate(corpus.FAMILIES):
        for family_case in range(20):
            index = offset + family_index * 20 + family_case
            evidence = root / "artifacts" / f"{role}-{index}-evidence.json"
            solar = root / "artifacts" / f"{role}-{index}-solar.yaml"
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(f"evidence-{index}\n", encoding="utf-8")
            solar.write_text(f"solar-{index}\n", encoding="utf-8")
            cases.append(
                corpus.DiagnosticValidationCase(
                    case_id=f"{role}-{index}",
                    pair_id=f"{index + 1:064x}",
                    workload_kind=family,
                    evidence_manifest=corpus.ValidationArtifactReference(
                        path=evidence.relative_to(root).as_posix(),
                        sha256=corpus.sha256_file(evidence),
                        size_bytes=evidence.stat().st_size,
                    ),
                    solar_manifest=corpus.ValidationArtifactReference(
                        path=solar.relative_to(root).as_posix(),
                        sha256=corpus.sha256_file(solar),
                        size_bytes=solar.stat().st_size,
                    ),
                )
            )
    source = corpus.DiagnosticValidationCorpus(role=role, cases=cases)
    path = root / f"{role}.json"
    corpus.atomic_write_json_value(path, source.model_dump(mode="json"))
    return path


def test_p0_conformance_currentizes_tree_references(
    load_script,
    tmp_path: Path,
) -> None:
    conformance = load_script(
        "scripts/internal/build_diagnostic_p0_conformance.py",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("evidence\n", encoding="utf-8")
    reference = {
        "path": artifact.name,
        "sha256": conformance.sha256_file(artifact),
    }
    case = {
        "evidence_manifest": dict(reference),
        "solar_manifest": dict(reference),
    }

    conformance._currentize_case_references(case, tmp_path)

    assert case["evidence_manifest"]["blob_backed"] is False
    assert case["evidence_manifest"]["size_bytes"] == artifact.stat().st_size


def test_p0_conformance_rebinds_currentized_calibration_audit(
    load_script,
    tmp_path: Path,
) -> None:
    conformance = load_script(
        "scripts/internal/build_diagnostic_p0_conformance.py",
    )
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    audit = {
        "parameter_estimation_evidence": [{"batch": 1}],
        "tuning_evidence": [{"batch": 2}],
        "probe_identity": {"gpu": "gfx1200"},
    }
    conformance.atomic_write_json_value(
        calibration / "profile.audit.json", audit
    )
    conformance.atomic_write_json_value(
        calibration / "profile.json",
        {
            "parameter_estimation_evidence_sha256": ["stale"],
            "tuning_evidence_sha256": ["stale"],
            "probe_evidence_sha256": ["stale"],
        },
    )

    conformance._rebind_calibration_audit_hashes(calibration)

    profile = conformance._load_object(calibration / "profile.json")
    assert profile["parameter_estimation_evidence_sha256"] == [
        conformance.stable_json_checksum(
            audit["parameter_estimation_evidence"]
        ),
        conformance.sha256_file(calibration / "profile.audit.json"),
    ]
    assert profile["tuning_evidence_sha256"] == [
        conformance.stable_json_checksum(audit["tuning_evidence"])
    ]
    assert profile["probe_evidence_sha256"] == [
        conformance.stable_json_checksum(audit["probe_identity"])
    ]


def test_rdna4_diagnostic_promotion_verifies_roles_order_and_hashes(
    load_script,
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )
    monkeypatch.setenv(
        "SOL_EXECBENCH_DIAGNOSTIC_STORE",
        str(tmp_path / "store"),
    )
    development = _write_promotion_source(corpus, tmp_path, "development", 0)
    held_out = _write_promotion_source(corpus, tmp_path, "held_out", 220)
    output = tmp_path / "promoted-development.json"

    corpus._promote_development(
        tmp_path,
        [development, held_out],
        output,
    )

    promoted = corpus.load_json_file(corpus.DiagnosticValidationCorpus, output)
    assert promoted.role == "development"
    assert len(promoted.cases) == 440
    assert promoted.cases[0].case_id.startswith("promoted-00-")
    assert promoted.cases[-1].case_id.startswith("promoted-01-")
    # Promotion emits content-addressed blob references, not path trees.
    for case in promoted.cases:
        assert case.evidence_manifest.blob_backed is True
        assert case.solar_manifest.blob_backed is True
    with pytest.raises(ValueError, match="order"):
        corpus._promote_development(
            tmp_path,
            [held_out, development],
            tmp_path / "wrong-order.json",
        )
    with pytest.raises(ValueError, match="already exists"):
        corpus._promote_development(
            tmp_path,
            [development, held_out],
            output,
        )


def test_rdna4_diagnostic_promotion_rejects_hash_drift(
    load_script,
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )
    monkeypatch.setenv(
        "SOL_EXECBENCH_DIAGNOSTIC_STORE",
        str(tmp_path / "store"),
    )
    development = _write_promotion_source(corpus, tmp_path, "development", 0)
    held_out = _write_promotion_source(corpus, tmp_path, "held_out", 220)
    (tmp_path / "artifacts/development-0-evidence.json").write_text(
        "changed\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        corpus._promote_development(
            tmp_path,
            [development, held_out],
            tmp_path / "promoted-development.json",
        )


def test_rdna4_diagnostic_promotion_independent_of_historical_paths(
    load_script,
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )
    monkeypatch.setenv(
        "SOL_EXECBENCH_DIAGNOSTIC_STORE",
        str(tmp_path / "store"),
    )
    output_root = tmp_path / "outputs"
    development_root = output_root / "cycle1"
    held_out_root = output_root / "cycle2"
    development = _write_promotion_source(
        corpus, development_root, "development", 0
    )
    held_out = _write_promotion_source(corpus, held_out_root, "held_out", 220)
    output = output_root / "promoted-development.json"

    corpus._promote_development(
        output_root,
        [development, held_out],
        output,
    )

    promoted = corpus.load_json_file(corpus.DiagnosticValidationCorpus, output)
    for case in promoted.cases:
        for reference in (case.evidence_manifest, case.solar_manifest):
            # Blob references carry no path and depend on no historical tree.
            assert reference.blob_backed is True
            store = corpus.BlobStore(tmp_path / "store")
            resolved = store.get(reference.sha256)
            assert resolved.stat().st_size == reference.size_bytes
            assert corpus.sha256_file(resolved) == reference.sha256


def test_rdna4_diagnostic_promotion_rejects_source_outside_root(
    load_script,
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )
    monkeypatch.setenv(
        "SOL_EXECBENCH_DIAGNOSTIC_STORE",
        str(tmp_path / "store"),
    )
    output_root = tmp_path / "outputs"
    development = _write_promotion_source(
        corpus, tmp_path / "outside", "development", 0
    )
    held_out = _write_promotion_source(
        corpus, output_root / "cycle2", "held_out", 220
    )

    with pytest.raises(ValueError, match="remain under --root"):
        corpus._promote_development(
            output_root,
            [development, held_out],
            output_root / "promoted-development.json",
        )


def test_rdna4_diagnostic_promotion_rejects_artifact_outside_source_root(
    load_script,
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py",
    )
    monkeypatch.setenv(
        "SOL_EXECBENCH_DIAGNOSTIC_STORE",
        str(tmp_path / "store"),
    )
    output_root = tmp_path / "outputs"
    development_root = output_root / "cycle1"
    development = _write_promotion_source(
        corpus, development_root, "development", 0
    )
    held_out = _write_promotion_source(
        corpus, output_root / "cycle2", "held_out", 220
    )
    escaped = output_root / "escaped-evidence.json"
    escaped.write_text("evidence-0\n", encoding="utf-8")
    evidence = development_root / "artifacts/development-0-evidence.json"
    evidence.unlink()
    evidence.symlink_to(escaped)

    with pytest.raises(ValueError, match="escapes its corpus root"):
        corpus._promote_development(
            output_root,
            [development, held_out],
            output_root / "promoted-development.json",
        )


def _result_output(values: Sequence[object]) -> str:
    return "\n".join(f"RESULT {value}" for value in values)


def test_resource_peak_calibration_parses_and_summarizes_samples(
    load_script,
) -> None:
    calibration = load_script(
        "scripts/internal/rdna4/run_rdna4_resource_peak_calibration.py",
    )
    values = [1, 2, 3, 4, 5, 6, 7]
    samples = calibration._parse_result_samples(
        Path("probe"),
        _result_output(values),
    )
    batches = (
        calibration.SampleBatch(
            process_batch=0,
            samples=samples,
            telemetry_before={"gfx_clock_mhz": 1000, "deep_sleep": "disabled"},
            telemetry_after={"gfx_clock_mhz": 1200, "deep_sleep": "disabled"},
        ),
    )

    assert calibration._nested_value({"a": {"b": 3}}, "a", "b") == 3
    assert calibration._nested_value({"a": 1}, "a", "b") is None
    assert calibration._metric_value({"a": {"value": 2}}, "a") == 2.0
    assert calibration._metric_value({"a": True}, "a") is None
    assert calibration._flatten_samples(batches) == samples
    assert calibration._numeric_summary(samples) == {
        "minimum": 1.0,
        "median": 4.0,
        "maximum": 7.0,
    }
    assert calibration._numeric_summary(()) is None
    assert calibration._sample_statistics(batches)["primary_result"] == 4.0
    assert calibration._telemetry_summary(batches)["numeric"][
        "gfx_clock_mhz"
    ] == {
        "minimum": 1000.0,
        "median": 1100.0,
        "maximum": 1200.0,
    }
    assert batches[0].to_dict()["median"] == 4


@pytest.mark.parametrize(
    "values",
    (
        [1, 2],
        [1, 2, 3, 4, 5, 6, "invalid"],
        [1, 2, 3, 4, 5, 6, 0],
        [1, 2, 3, 4, 5, 6, "nan"],
    ),
)
def test_resource_peak_calibration_rejects_invalid_sample_batches(
    load_script,
    values: list[object],
) -> None:
    calibration = load_script(
        "scripts/internal/rdna4/run_rdna4_resource_peak_calibration.py",
    )

    with pytest.raises(RuntimeError, match="produced"):
        calibration._parse_result_samples(Path("probe"), _result_output(values))

    with pytest.raises(RuntimeError, match="contain no RESULT"):
        calibration._flatten_samples(())


def test_resource_peak_calibration_requires_complete_coverage(
    load_script,
) -> None:
    calibration = load_script(
        "scripts/internal/rdna4/run_rdna4_resource_peak_calibration.py",
    )
    measurements = [
        {
            "covers_precisions": probe["covers_precisions"],
            "covers_resource_modes": probe["covers_resource_modes"],
        }
        for probe in calibration.PROBES
    ]

    assert calibration._calibration_coverage(measurements)["status"] == "passed"
    with pytest.raises(RuntimeError, match="do not exactly cover"):
        calibration._calibration_coverage(
            [{"covers_precisions": (), "covers_resource_modes": ()}],
        )


def test_rdna4_validation_helpers_and_verify_mode(
    load_script,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    validation = load_script("scripts/internal/rdna4/run_rdna4_validation.py")
    output = validation._prepare_output(tmp_path / "new")
    monkeypatch.setattr(
        validation,
        "verify_validation_directory",
        lambda *a, **k: None,
    )
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    assert output.is_dir()
    assert validation._attestation() == {
        "kind": "local_unsigned",
        "trusted_execution": False,
    }
    assert validation.main(["--verify", str(output)]) == 0
    assert capsys.readouterr().out.strip().endswith("manifest.json")

    (output / "existing").write_text("occupied", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        validation._prepare_output(output)


def test_rdna4_validation_timeout_and_argument_checks(
    load_script,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = load_script("scripts/internal/rdna4/run_rdna4_validation.py")

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(["pytest"], 1)

    monkeypatch.setattr(
        validation,
        "run_in_process_group_to_files",
        raise_timeout,
    )

    assert validation._run_tests(tmp_path, 1.0) == 124
    with pytest.raises(ValueError, match="timeout must be positive"):
        validation.main(
            ["--output-dir", str(tmp_path / "out"), "--timeout", "0"],
        )


def test_force_refuses_to_recollect_frozen_held_out(
    tmp_path: Path,
    load_script: Any,
) -> None:
    """Gap 5: --force cannot silently overwrite a frozen held-out corpus."""
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
    )
    (tmp_path / "held_out.json").write_text("{}", encoding="utf-8")

    forced = SimpleNamespace(
        stage="collect",
        role="held_out",
        force=True,
        root=tmp_path,
        confirm_recollect_held_out=False,
    )
    with pytest.raises(ValueError, match="frozen held-out corpus"):
        corpus._refuse_frozen_held_out_recollect(forced)


def test_force_recollect_held_out_cannot_be_confirmed(
    tmp_path: Path,
    load_script: Any,
) -> None:
    """Gap 1: no confirmation flag can mutate a frozen held-out generation."""
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
    )
    (tmp_path / "held_out.json").write_text("{}", encoding="utf-8")

    confirmed = SimpleNamespace(
        stage="collect",
        role="held_out",
        force=True,
        root=tmp_path,
    )
    with pytest.raises(ValueError, match="frozen held-out corpus"):
        corpus._refuse_frozen_held_out_recollect(confirmed)


def test_solar_force_refuses_frozen_held_out(
    tmp_path: Path,
    load_script: Any,
) -> None:
    """Gap 1: solar --force cannot rewrite a frozen held-out generation."""
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
    )
    (tmp_path / "held_out.json").write_text("{}", encoding="utf-8")

    forced = SimpleNamespace(
        stage="solar",
        role="held_out",
        force=True,
        root=tmp_path,
    )
    with pytest.raises(ValueError, match="frozen held-out corpus"):
        corpus._refuse_frozen_held_out_recollect(forced)


def test_freeze_refuses_to_overwrite_a_frozen_corpus(
    tmp_path: Path,
    load_script: Any,
) -> None:
    """Gap 1: freeze never overwrites an existing frozen corpus file."""
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
    )
    (tmp_path / "held_out.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        corpus._freeze(tmp_path, "held_out")


def test_force_held_out_without_frozen_corpus_is_allowed(
    tmp_path: Path,
    load_script: Any,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
    )

    fresh = SimpleNamespace(
        stage="collect",
        role="held_out",
        force=True,
        root=tmp_path,
        confirm_recollect_held_out=False,
    )
    corpus._refuse_frozen_held_out_recollect(fresh)  # must not raise
