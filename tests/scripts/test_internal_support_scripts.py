from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


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


def test_orojenesis_provenance_hash_and_compiler_identity(
    load_script,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = load_script("scripts/internal/orojenesis/write_provenance.py")
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"mapper")
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="clang version 19.0\nCopyright\n",
        ),
    )

    assert provenance._sha256(artifact) == hashlib.sha256(b"mapper").hexdigest()
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
