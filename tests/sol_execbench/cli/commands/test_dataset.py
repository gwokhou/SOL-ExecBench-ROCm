from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from sol_execbench.cli.commands import dataset as cli_dataset
from sol_execbench.cli.main import cli
from sol_execbench.core.dataset.aka_compatibility import (
    AKAMaterializationTarget,
)
from sol_execbench.core.platform.runtime import RocmDeviceInfo


def _device(gfx_target: str = "gfx1150") -> RocmDeviceInfo:
    return RocmDeviceInfo(
        device="cuda:0",
        index=0,
        name="test GPU",
        gfx_target=gfx_target,
        visible_compute_units=32,
        total_memory_bytes=32 * 1024**3,
        l2_cache_bytes=16 * 1024**2,
        torch_version="test",
        hip_version="test",
    )


@pytest.mark.usefixtures("native_gpu_cpu_observation")
def test_materialize_detects_target_and_uses_target_specific_default(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    def materialize(output, *, target, probe_timeout_seconds):
        observed.update(
            output=output,
            target=target,
            probe_timeout_seconds=probe_timeout_seconds,
        )
        return output

    manifest = SimpleNamespace(
        materialize=materialize,
        audit=lambda _output: {
            "status": "valid",
            "problems": 2,
            "workloads": 3,
            "excluded_workloads": 1,
            "gfx_target": "gfx1150",
        },
    )
    monkeypatch.setattr(
        cli_dataset.AKACorpusManifest,
        "load",
        lambda _path: manifest,
    )
    monkeypatch.setattr(
        cli_dataset,
        "detect_rocm_device",
        lambda device: _device(),
    )

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "dataset",
            "materialize",
            "--skip-aka-fetch",
            "--device",
            "cuda:0",
            "--target-arch",
            "gfx1150",
        ],
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.output)
    assert response["data"]["gfx_target"] == "gfx1150"
    assert str(observed["output"]) == "problems/local/AMD_AKA/gfx1150"
    assert observed["probe_timeout_seconds"] == 120.0
    target = observed["target"]
    assert isinstance(target, AKAMaterializationTarget)
    assert target.cache_clear.clear_buffer_bytes == 32 * 1024**2


@pytest.mark.usefixtures("native_gpu_cpu_observation")
def test_materialize_rejects_detected_target_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_dataset.AKACorpusManifest,
        "load",
        lambda _path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        cli_dataset,
        "detect_rocm_device",
        lambda _device_name: _device("gfx1200"),
    )

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "dataset",
            "materialize",
            "--skip-aka-fetch",
            "--target-arch",
            "gfx942",
        ],
    )

    assert result.exit_code == 2
    response = json.loads(result.output)
    assert response["error"]["code"] == "aka_target_mismatch"


def test_audit_cli_translates_provenance_and_yaml_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bad_manifest = tmp_path / "bad.yaml"
    bad_manifest.write_text("a: [unterminated\n", encoding="utf-8")
    problem_root = tmp_path / "problems"
    problem_root.mkdir()

    yaml_result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "dataset",
            "audit",
            "--manifest",
            str(bad_manifest),
            str(problem_root),
        ],
    )
    assert yaml_result.exit_code == 2
    assert json.loads(yaml_result.output)["error"]["code"] == (
        "invalid_aka_manifest"
    )

    aka_root = tmp_path / "aka"
    aka_root.mkdir()
    manifest = SimpleNamespace(
        audit=lambda _root: {"problems": 1, "scored": 1},
        audit_aka_provenance=lambda _aka: (_ for _ in ()).throw(
            ValueError("AKA clone missing .aka-head")
        ),
    )
    monkeypatch.setattr(
        cli_dataset.AKACorpusManifest,
        "load",
        lambda _path: manifest,
    )
    prov_result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "dataset",
            "audit",
            "--aka-root",
            str(aka_root),
            str(problem_root),
        ],
    )
    assert prov_result.exit_code == 2
    assert json.loads(prov_result.output)["error"]["code"] == "aka_audit_failed"


def test_ensure_aka_clone_passes_aka_root_and_captures_stdout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    aka_root = tmp_path / "custom-aka"
    observed_argv: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        observed_argv.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="cloned\n", stderr="")

    monkeypatch.setattr(cli_dataset, "run_in_process_group_bounded", fake_run)
    cli_dataset._ensure_aka_clone(aka_root)

    assert len(observed_argv) == 1
    assert observed_argv[0][-1] == str(aka_root)
