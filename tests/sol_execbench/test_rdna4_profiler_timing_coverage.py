from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from sol_execbench.core.dataset import (
    build_dataset_inventory,
    build_profiler_timing_coverage_report,
    classify_rocm_readiness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/run_rdna4_profiler_timing_coverage.py"
SPEC = spec_from_file_location("run_rdna4_profiler_timing_coverage", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
coverage_script = module_from_spec(SPEC)
sys.modules[SPEC.name] = coverage_script
SPEC.loader.exec_module(coverage_script)


def _write_problem(root: Path, name: str) -> None:
    problem_dir = root / "L1" / name
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text(
        json.dumps(
            {
                "name": name,
                "description": "forward demo",
                "axes": {"N": {"type": "var"}},
                "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
                "outputs": {"out": {"shape": ["N"], "dtype": "float32"}},
                "reference": "def run(x):\n    return x\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (problem_dir / "workload.jsonl").write_text(
        json.dumps(
            {
                "uuid": "w0",
                "axes": {"N": 4},
                "inputs": {"x": {"type": "random"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (problem_dir / "reference.py").write_text(
        "def run(x):\n    return x\n",
        encoding="utf-8",
    )


def test_blocker_ledger_records_non_passing_rows(tmp_path):
    dataset_root = tmp_path / "dataset"
    timing_root = tmp_path / "timing"
    _write_problem(dataset_root, "passing")
    _write_problem(dataset_root, "fallback")
    (timing_root / "L1").mkdir(parents=True)
    (timing_root / "L1" / "passing.timing.json").write_text(
        json.dumps(
            {
                "profiler_collected": True,
                "selection": {"policy": {"backend": "rocprofv3"}},
                "evidence": {
                    "backend": "rocprofv3",
                    "parsed_rows": [{"is_kernel_activity": True}],
                },
                "replacement_metadata": {"full_workload_coverage": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (timing_root / "L1" / "fallback.timing.json").write_text(
        json.dumps(
            {
                "profiler_collected": False,
                "selection": {
                    "reason": "selected policy backend is device_events",
                    "policy": {"backend": "device_events"},
                },
                "evidence": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    inventory = build_dataset_inventory(dataset_root, categories=("L1",))
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)
    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=(timing_root,),
        expected_problem_denominator=2,
        created_at="2026-06-08T00:00:00Z",
    )

    ledger = coverage_script.build_blocker_ledger(report)

    assert ledger["schema_version"] == "sol_execbench.rdna4_coverage_blocker_ledger.v1"
    assert ledger["blocked_or_non_passing_count"] == 1
    assert ledger["profiler_backed_problems"] == 1
    assert ledger["ledger_checksum"]
    assert ledger["rows"] == [
        {
            "problem_id": "L1/fallback",
            "category": "L1",
            "problem_path": "L1/fallback",
            "status": "timing_fallback",
            "readiness_status": "ready",
            "workload_count": 1,
            "readiness_reason_codes": ["ready_to_attempt_rocm_execution"],
            "readiness_blocker_types": [],
            "evidence_path": (timing_root / "L1" / "fallback.timing.json").as_posix(),
            "blocker_class": None,
            "trace_status_counts": {},
            "fallback_reason": "selected policy backend is device_events",
            "replacement_failure_reason": None,
        }
    ]
