"""Check host launch boundaries without starting a container."""

import json

import pytest

from scripts.internal.solar import e2_successor_host as host
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file


def _fixture(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    spec = {
        "image_id": "sha256:" + "a" * 64,
        "activity_directory": "frozen-study",
        "worker_memory_bytes": 16081639424,
        "cpu_affinity": [0, 1, 2],
    }
    path = bundle / "freeze.json"
    atomic_write_json_value(path, spec)
    opts = [
        "--repository",
        str(tmp_path),
        "--bundle",
        str(bundle),
        "--freeze-sha256",
        sha256_file(path),
        "--image-tag",
        "fixture",
    ]
    return bundle, opts, spec


def test_plan_derives_device_free_postflight_from_the_same_freeze(
    tmp_path, monkeypatch, capsys
):
    def fail(*_args, **_kwargs):
        pytest.fail("plan must never spawn a host process")

    monkeypatch.setattr(host, "run_in_process_group_bounded", fail)
    bundle, opts, spec = _fixture(tmp_path)
    assert host.main(["plan", *opts]) == 0
    plan = json.loads(capsys.readouterr().out)["commands"]
    assert set(plan) == {"online", "offline", "audit", "report"}
    assert plan["online"][0] == "bash"
    assert "--open-holdout" in plan["online"]
    assert plan["online"].count("--init") == 1
    assert plan["online"].index("--init") < plan["online"].index("--")
    for phase in ("offline", "audit", "report"):
        command = plan[phase]
        assert command[:2] == ["docker", "run"]
        assert not any(a.startswith(("--device", "--gpus")) for a in command)
        assert "--open-holdout" not in command
        assert "--network=none" in command
        assert spec["image_id"] in command
        assert "--cpuset-cpus=0,1,2" in command
    assert not (bundle / "frozen-study").exists()
    assert not (bundle / "host-run").exists()


def test_host_run_demands_authorization_before_any_io(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        pytest.fail("unauthorized run must not read a freeze or launch")

    monkeypatch.setattr(host, "sha256_file", fail)
    monkeypatch.setattr(host, "_run_activity", fail)
    with pytest.raises(SystemExit, match="2"):
        host.main(
            [
                "run",
                "--repository",
                str(tmp_path),
                "--bundle",
                str(tmp_path / "absent"),
                "--freeze-sha256",
                "b" * 64,
                "--image-tag",
                "fixture",
            ]
        )


def test_host_run_rejects_command_drift_before_launch(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        pytest.fail("changed host plan must stop before launch")

    _, opts, _ = _fixture(tmp_path)
    plan = tmp_path / "host-plan.json"
    atomic_write_json_value(plan, {})
    monkeypatch.setattr(host, "_run_activity", fail)
    with pytest.raises(ValueError, match="differs from its predeclared plan"):
        host.main(["run", *opts, "--open-holdout", "--host-plan", str(plan)])
