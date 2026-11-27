"""CLI integration tests for ``sol-execbench vertical`` subcommands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.vertical.report import compare_summaries_markdown


def test_vertical_cli_help_and_cpu_smoke_roundtrip(tmp_path: Path) -> None:
    runner = CliRunner()
    help_res = runner.invoke(cli, ["vertical", "--help"])
    assert help_res.exit_code == 0, help_res.output
    assert {"run", "compare", "preflight"}.issubset(
        set(help_res.output.split())
    )

    pf_help = runner.invoke(cli, ["vertical", "preflight", "--help"])
    assert pf_help.exit_code == 0, pf_help.output
    assert {"twin", "probe", "bundle", "verify"}.issubset(
        set(pf_help.output.split())
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

    # A bundle containing failed/unavailable workloads must exit with code 1.
    unavail_dir = tmp_path / "aiter_unavail"
    unavail_run = runner.invoke(
        cli,
        [
            "vertical",
            "run",
            "--mode",
            "smoke",
            "--device",
            "cpu",
            "--backend",
            "aiter",
            "--target-override",
            "MI300X",
            "--output-dir",
            str(unavail_dir),
        ],
    )
    assert unavail_run.exit_code == 0, unavail_run.output
    failed_verify = runner.invoke(
        cli,
        [
            "--format",
            "json",
            "vertical",
            "preflight",
            "verify",
            str(unavail_dir),
        ],
    )
    assert failed_verify.exit_code == 1, failed_verify.output
    failed_payload = json.loads(failed_verify.output)
    assert failed_payload["data"]["all_passed"] is False

    # Missing workload_id in JSON summary must fail closed.
    bad_summary = tmp_path / "bad_summary.json"
    bad_summary.write_text(
        json.dumps(
            {
                "target": {"target_id": "MI300X"},
                "backend": "portable",
                "rows": [{"implementation": "portable_triton"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing workload_id"):
        compare_summaries_markdown(pt_dir / "summary.json", bad_summary)
