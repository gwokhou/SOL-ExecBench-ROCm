from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from sol_execbench.cli.main import cli

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "problems/LLM_CORE/releases/LLM_CORE_V1/manifest.yaml"
TARGET = ROOT / "problems/LLM_CORE/targets/gfx1200.yaml"


def test_corpus_validate_cli_reports_frozen_release() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "dataset",
            "corpus",
            "validate",
            "--manifest",
            str(MANIFEST),
        ],
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.output)
    assert response["data"]["release_id"] == "LLM_CORE_V1"
    assert response["data"]["definitions"] == 84
    assert response["data"]["workloads"] == 1260


def test_corpus_select_cli_never_probes_hardware(tmp_path: Path) -> None:
    output = tmp_path / "selection"
    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "dataset",
            "corpus",
            "select",
            "--manifest",
            str(MANIFEST),
            "--target-descriptor",
            str(TARGET),
            "--memory-budget",
            str(8 * 1024**3),
            "--profile",
            "core",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.output)
    assert response["data"]["qualification_status"] == "declared"
    record = yaml.safe_load(
        (output / "selection-manifest.yaml").read_text(encoding="utf-8"),
    )
    assert record["target"]["memory_budget_bytes"] == 8 * 1024**3
    assert record["coverage"]["workloads"] > 0


def test_corpus_select_cli_requires_one_target_source(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "dataset",
            "corpus",
            "select",
            "--manifest",
            str(MANIFEST),
            "--memory-budget",
            "1024",
            "--output",
            str(tmp_path / "selection"),
        ],
    )

    assert result.exit_code == 2
    response = json.loads(result.output)
    assert response["error"]["code"] == "invalid_static_target"
