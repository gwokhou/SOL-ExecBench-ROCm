from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts/check_dataset_redistribution.py"
SPEC = spec_from_file_location("check_dataset_redistribution", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_dataset_redistribution = module_from_spec(SPEC)
sys.modules[SPEC.name] = check_dataset_redistribution
SPEC.loader.exec_module(check_dataset_redistribution)


def _policy() -> dict[str, object]:
    return check_dataset_redistribution.load_dataset_policy(
        REPO_ROOT / "provenance.toml"
    )


def test_blocks_nvidia_dataset_paths_for_repository_redistribution() -> None:
    findings = check_dataset_redistribution.check_paths(
        [
            "data/SOL-ExecBench/benchmark/L1/problem/definition.json",
            "src/sol_execbench/core/dataset/manifest.py",
        ],
        _policy(),
        mode="repository",
    )

    assert len(findings) == 1
    assert findings[0].source_id == "nvidia_sol_execbench"
    assert findings[0].redistribution_class == "excluded"


def test_blocks_generated_nvidia_derivatives_for_release_bundles(tmp_path) -> None:
    restricted = tmp_path / "out/dataset_migration/nvidia-sol-execbench/problem.json"
    restricted.parent.mkdir(parents=True)
    restricted.write_text(
        json.dumps({"source_boundary": "nvidia_sol_execbench"}), encoding="utf-8"
    )
    allowed = tmp_path / "docs/provenance.md"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("# Provenance\n", encoding="utf-8")

    findings = check_dataset_redistribution.check_release_root(tmp_path, _policy())

    assert len(findings) == 1
    assert findings[0].path == "out/dataset_migration/nvidia-sol-execbench/problem.json"
    assert findings[0].source_id == "nvidia_sol_execbench"


def test_allows_project_code_and_flashinfer_apache_paths() -> None:
    findings = check_dataset_redistribution.check_paths(
        [
            "src/sol_execbench/core/dataset/manifest.py",
            "third_party/flashinfer-trace/NOTICE",
        ],
        _policy(),
        mode="repository",
    )

    assert findings == []


def test_cli_json_reports_blocking_status_for_restricted_path(capsys) -> None:
    status = check_dataset_redistribution.main(
        [
            "--path",
            "data/SOL-ExecBench/benchmark/L1/problem/workload.jsonl",
            "--json",
        ]
    )

    assert status == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_status"] == "blocking"
    assert payload["findings"][0]["source_id"] == "nvidia_sol_execbench"
