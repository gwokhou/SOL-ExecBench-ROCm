from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_derived_isolated.py"
spec = importlib.util.spec_from_file_location("run_derived_isolated", SCRIPT_PATH)
assert spec is not None
run_derived_isolated = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = run_derived_isolated
spec.loader.exec_module(run_derived_isolated)


def _write_problem(dataset_root: Path, category: str, name: str) -> Path:
    problem_dir = dataset_root / category / name
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(
        json.dumps({"name": name, "reference": "def run(x):\n    return x\n"})
    )
    (problem_dir / "workload.jsonl").write_text("{}\n")
    return problem_dir


def test_wrap_command_uses_systemd_memory_properties():
    wrapped = run_derived_isolated.wrap_command(
        ["uv", "run", "scripts/run_dataset.py"],
        launch_mode="systemd",
        memory_max="20G",
        memory_swap_max="0",
        unit_name="sol-derived-demo",
    )

    assert wrapped[:4] == ["systemd-run", "--user", "--wait", "--collect"]
    assert "--property" in wrapped
    assert "MemoryMax=20G" in wrapped
    assert "MemorySwapMax=0" in wrapped
    assert wrapped[-3:] == ["uv", "run", "scripts/run_dataset.py"]


def test_resume_skips_successful_status_entries(tmp_path):
    status_path = tmp_path / "status.jsonl"
    status_path.write_text(
        json.dumps(
            {
                "problem_id": "L1/a",
                "status": "ok",
                "returncode": 0,
            }
        )
        + "\n"
        + json.dumps(
            {
                "problem_id": "L1/b",
                "status": "failed",
                "returncode": -9,
            }
        )
        + "\n"
    )

    assert run_derived_isolated.load_completed(status_path) == {"L1/a"}
    assert run_derived_isolated.should_skip(
        "L1/a", completed={"L1/a"}, start_at=None, start_after=None
    )
    assert not run_derived_isolated.should_skip(
        "L1/b", completed={"L1/a"}, start_at=None, start_after=None
    )


def test_main_records_per_problem_status_and_continues(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    first = _write_problem(dataset_root, "L1", "a")
    second = _write_problem(dataset_root, "L1", "b")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=0 if str(first) in command else 7,
        )

    monkeypatch.setattr(run_derived_isolated.subprocess, "run", fake_run)

    rc = run_derived_isolated.main(
        [
            str(dataset_root),
            "-o",
            str(tmp_path / "out"),
            "--category",
            "L1",
            "--amd-sol-bound-dir",
            str(tmp_path / "sol"),
            "--solar-derivation",
            str(tmp_path / "solar"),
            "--status-jsonl",
            str(tmp_path / "status.jsonl"),
            "--log-file",
            str(tmp_path / "derived.log"),
            "--continue-on-failure",
        ]
    )

    assert rc == 1
    assert len(calls) == 2
    statuses = [
        json.loads(line)
        for line in (tmp_path / "status.jsonl").read_text().splitlines()
    ]
    assert [item["problem_id"] for item in statuses] == ["L1/a", "L1/b"]
    assert [item["status"] for item in statuses] == ["ok", "failed"]
    assert str(first) in statuses[0]["command"]
    assert str(second) in statuses[1]["command"]
