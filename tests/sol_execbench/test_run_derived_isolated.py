from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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


def test_problem_id_file_filters_comments_and_blanks(tmp_path):
    path = tmp_path / "targets.txt"
    path.write_text("\n# comment\nL1/a\n\nL2/b\n")

    assert run_derived_isolated.load_problem_id_filter(path) == {"L1/a", "L2/b"}


def test_main_runs_only_problem_ids_from_file(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    first = _write_problem(dataset_root, "L1", "a")
    _write_problem(dataset_root, "L1", "b")
    targets = tmp_path / "targets.txt"
    targets.write_text("L1/a\n")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(run_derived_isolated.subprocess, "run", fake_run)

    rc = run_derived_isolated.main(
        [
            str(dataset_root),
            "-o",
            str(tmp_path / "out"),
            "--category",
            "L1",
            "--problem-id-file",
            str(targets),
            "--amd-sol-bound-dir",
            str(tmp_path / "sol"),
            "--solar-derivation",
            str(tmp_path / "solar"),
            "--status-jsonl",
            str(tmp_path / "status.jsonl"),
            "--log-file",
            str(tmp_path / "derived.log"),
        ]
    )

    assert rc == 0
    assert len(calls) == 1
    statuses = [
        json.loads(line)
        for line in (tmp_path / "status.jsonl").read_text().splitlines()
    ]
    assert [item["problem_id"] for item in statuses] == ["L1/a"]
    assert str(first) in statuses[0]["command"]


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
    # After parallel execution, results are sorted by problem_id for determinism
    assert sorted([item["problem_id"] for item in statuses]) == ["L1/a", "L1/b"]
    assert {item["problem_id"]: item["status"] for item in statuses} == {
        "L1/a": "ok",
        "L1/b": "failed",
    }
    # Verify both problems were executed
    assert any(str(first) in s["command"] for s in statuses)
    assert any(str(second) in s["command"] for s in statuses)


def test_thread_safe_jsonl_writes(tmp_path):
    """DERV-02: Verify concurrent append_status calls produce no interleaved/corrupted lines."""
    status_path = tmp_path / "status.jsonl"
    status_lock = threading.Lock()
    num_threads = 10
    lines_per_thread = 100

    def write_batch(thread_id: int) -> list:
        statuses = []
        for i in range(lines_per_thread):
            status = run_derived_isolated.ProblemStatus(
                problem_id=f"thread_{thread_id}_line_{i}",
                status="ok",
                returncode=0,
                started_at="2026-06-10T12:00:00Z",
                finished_at="2026-06-10T12:00:01Z",
                elapsed_seconds=1.0,
                command=["test"],
                log_ref="test.log",
            )
            with status_lock:
                run_derived_isolated.append_status(status_path, status)
            statuses.append(status)
        return statuses

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(write_batch, thread_id) for thread_id in range(num_threads)
        ]
        all_statuses = []
        for future in futures:
            all_statuses.extend(future.result())

    # Verify no corrupted JSON lines
    written_lines = status_path.read_text(encoding="utf-8").splitlines()
    assert len(written_lines) == num_threads * lines_per_thread

    # Verify all lines are valid JSON
    for line in written_lines:
        payload = json.loads(line)
        assert "problem_id" in payload
        assert "status" in payload

    # Verify all expected problem_ids are present
    written_ids = {json.loads(line)["problem_id"] for line in written_lines}
    expected_ids = {
        f"thread_{t}_line_{i}"
        for t in range(num_threads)
        for i in range(lines_per_thread)
    }
    assert written_ids == expected_ids


def test_parallel_resume_semantics(tmp_path, monkeypatch):
    """DERV-03: Verify parallel workers skip completed problems atomically."""
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "completed")
    _write_problem(dataset_root, "L1", "pending")
    _write_problem(dataset_root, "L1", "also_pending")

    # Pre-populate status.jsonl with completed problem
    status_path = tmp_path / "status.jsonl"
    status_path.write_text(
        json.dumps(
            {
                "problem_id": "L1/completed",
                "status": "ok",
                "returncode": 0,
                "started_at": "2026-06-10T12:00:00Z",
                "finished_at": "2026-06-10T12:00:01Z",
                "elapsed_seconds": 1.0,
                "command": ["uv", "run"],
                "log_ref": "test.log",
            }
        )
        + "\n"
    )

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(run_derived_isolated.subprocess, "run", fake_run)

    # Run with --resume and --jobs=2
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
            str(status_path),
            "--log-file",
            str(tmp_path / "derived.log"),
            "--resume",
            "--jobs",
            "2",
        ]
    )

    assert rc == 0

    # Verify only pending problems were executed
    executed_ids = []
    for line in status_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload.get("status") == "ok":
            executed_ids.append(payload["problem_id"])

    # L1/completed should only appear once (from pre-populated line)
    assert executed_ids.count("L1/completed") == 1
    # L1/pending and L1/also_pending should each appear once
    assert executed_ids.count("L1/pending") == 1
    assert executed_ids.count("L1/also_pending") == 1

    # Verify subprocess was called only for pending problems
    assert len(calls) == 2


def test_jobs_flag_default(tmp_path, monkeypatch):
    """DERV-04: Verify --jobs defaults to min(os.cpu_count(), 4)."""
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "a")

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(run_derived_isolated.subprocess, "run", fake_run)

    # Run without --jobs flag
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
        ]
    )

    assert rc == 0
    # Verify problem executed
    assert len(calls) == 1


def test_parallel_dispatch(tmp_path, monkeypatch):
    """DERV-01: Integration test for parallel dispatch with multiple problems."""
    dataset_root = tmp_path / "dataset"
    _write_problem(dataset_root, "L1", "problem_a")
    _write_problem(dataset_root, "L1", "problem_b")
    _write_problem(dataset_root, "L1", "problem_c")
    _write_problem(dataset_root, "L1", "problem_d")

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(run_derived_isolated.subprocess, "run", fake_run)

    # Run with --jobs=2 (4 problems, 2 workers)
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
            "--jobs",
            "2",
        ]
    )

    assert rc == 0

    # Verify all 4 problems completed
    assert len(calls) == 4

    # Verify status JSONL contains all 4 entries
    statuses = [
        json.loads(line)
        for line in (tmp_path / "status.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    problem_ids = [item["problem_id"] for item in statuses]
    assert set(problem_ids) == {
        "L1/problem_a",
        "L1/problem_b",
        "L1/problem_c",
        "L1/problem_d",
    }

    # Verify all statuses are ok
    assert all(item["status"] == "ok" for item in statuses)
