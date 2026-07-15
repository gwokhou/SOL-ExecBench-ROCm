from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


SCRIPT_PATH = (
    Path(__file__).parents[4]
    / "scripts"
    / "internal"
    / "prepare_full_suite_baseline_inputs.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "prepare_full_suite_baseline_inputs", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace(uuid: str) -> dict[str, object]:
    return {
        "definition": "gemm",
        "workload": {"uuid": uuid},
        "evaluation": {
            "status": "PASSED",
            "performance": {"latency_ms": 1.0},
            "log": "Clocks locked: yes",
            "environment": {"hardware": "gfx1200"},
        },
    }


def test_filters_traces_that_are_explicitly_blocked_after_measurement(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_script()
    run_dir = tmp_path / "run"
    problem_dir = run_dir / "gemm"
    problem_dir.mkdir(parents=True)
    (problem_dir / "traces.json").write_text(
        json.dumps([_trace("eligible"), _trace("blocked")]), encoding="utf-8"
    )
    (problem_dir / "solution.json").write_text("{}\n", encoding="utf-8")
    coverage = tmp_path / "coverage.json"
    coverage.write_text('{"workloads": []}\n', encoding="utf-8")
    authority = tmp_path / "authority.json"
    authority.write_text(
        json.dumps(
            {
                "workloads": [
                    {
                        "definition": "gemm",
                        "workload_uuid": "eligible",
                        "official_blockers": [],
                    },
                    {
                        "definition": "gemm",
                        "workload_uuid": "blocked",
                        "official_blockers": ["sol_bound_sanity_failed"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            str(run_dir),
            str(coverage),
            str(output_dir),
            "--authority-input",
            str(authority),
            "--compiler-build-id",
            "hipcc-7.2-current-build",
        ],
    )

    assert module.main() == 0
    rows = [
        json.loads(line)
        for line in (output_dir / "baseline-trace.jsonl").read_text().splitlines()
    ]
    assert [(row["definition"], row["workload"]["uuid"]) for row in rows] == [
        ("gemm", "eligible")
    ]
    inputs = json.loads((output_dir / "baseline-inputs.json").read_text())
    assert inputs["compiler_build_id"] == "hipcc-7.2-current-build"
