from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from sol_execbench.cli.commands import dataset as cli_dataset
from sol_execbench.cli.main import cli
from sol_execbench.core.dataset.aka_compatibility import AkaMaterializationTarget
from sol_execbench.core.platform.runtime import RocmDeviceInfo


def _device(gfx_target: str = "gfx1150") -> RocmDeviceInfo:
    return RocmDeviceInfo(
        device="cuda:1",
        index=1,
        name="test GPU",
        gfx_target=gfx_target,
        total_memory_bytes=32 * 1024**3,
        l2_cache_bytes=16 * 1024**2,
        torch_version="test",
        hip_version="test",
    )


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
    monkeypatch.setattr(cli_dataset.AkaCorpusManifest, "load", lambda _path: manifest)
    monkeypatch.setattr(cli_dataset, "detect_rocm_device", lambda device: _device())

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "dataset",
            "materialize",
            "--skip-aka-fetch",
            "--device",
            "cuda:1",
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
    assert isinstance(target, AkaMaterializationTarget)
    assert target.cache_clear.clear_buffer_bytes == 32 * 1024**2


def test_materialize_rejects_detected_target_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_dataset.AkaCorpusManifest,
        "load",
        lambda _path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        cli_dataset, "detect_rocm_device", lambda _device_name: _device("gfx1200")
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
