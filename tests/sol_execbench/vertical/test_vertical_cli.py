"""CLI integration tests for ``sol-execbench vertical`` subcommands."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli.main import cli


def test_vertical_cli_help_and_cpu_smoke_roundtrip(tmp_path: Path) -> None:
    runner = CliRunner()
    help_res = runner.invoke(cli, ["vertical", "--help"])
    assert help_res.exit_code == 0, help_res.output
    assert {"run", "compare", "preflight"}.issubset(
        set(help_res.output.split())
    )

    pt_dir = tmp_path / "pytorch"
    port_dir = tmp_path / "portable"
    for backend, out_dir in (("pytorch", pt_dir), ("portable", port_dir)):
        run_res = runner.invoke(
            cli,
            [
                "--format",
                "json",
                "vertical",
                "run",
                "--mode",
                "smoke",
                "--device",
                "cpu",
                "--backend",
                backend,
                "--target-override",
                "MI300X",
                "--output-dir",
                str(out_dir),
            ],
        )
        assert run_res.exit_code == 0, run_res.output
        payload = json.loads(run_res.output)
        assert payload["ok"] is True
        assert Path(payload["data"]["summary_json"]).is_file()

    cmp_res = runner.invoke(
        cli,
        [
            "vertical",
            "compare",
            str(pt_dir / "summary.json"),
            str(port_dir / "summary.json"),
        ],
    )
    assert cmp_res.exit_code == 0, cmp_res.output
    assert "ROCm Vertical Stack SOL Diagnosis — MI300X" in cmp_res.output

    verify_res = runner.invoke(
        cli,
        ["--format", "json", "vertical", "preflight", "verify", str(pt_dir)],
    )
    assert verify_res.exit_code == 0, verify_res.output
    verify_payload = json.loads(verify_res.output)
    assert verify_payload["data"]["target_id"] == "MI300X"
    assert verify_payload["data"]["all_passed"] is True
