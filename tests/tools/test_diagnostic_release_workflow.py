from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"


def _workflows() -> dict[str, dict]:
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(WORKFLOW_DIR.glob("*.yml"))
    }


def test_self_hosted_runner_never_holds_release_authority() -> None:
    workflows = _workflows()
    hardware = workflows["rdna4-hardware.yml"]
    assert hardware["permissions"]["contents"] == "read"
    assert "write" not in hardware.get("permissions", {})


def test_only_hosted_release_jobs_hold_contents_write() -> None:
    workflows = _workflows()
    authorized = {"diagnostic-release.yml", "score-release.yml"}
    for name, definition in workflows.items():
        write = definition.get("permissions", {}).get("contents") == "write"
        assert write == (name in authorized), (
            f"{name} contents permission does not match release authority"
        )


def test_release_workflow_runs_on_github_hosted_runner() -> None:
    definition = _workflows()["diagnostic-release.yml"]
    job = next(iter(definition["jobs"].values()))
    assert "self-hosted" not in " ".join(job["runs-on"])


def test_release_workflow_publishes_with_the_fixed_attestation_asset() -> None:
    workflow = (WORKFLOW_DIR / "diagnostic-release.yml").read_text(
        encoding="utf-8"
    )

    assert (
        "$RUNNER_TEMP/release/"
        "diagnostic-lifecycle-p0-conformance-v1.attestation.json"
    ) in workflow
    assert '$RUNNER_TEMP/release/attestation.json"' not in workflow
    assert 'TAG_SHA="$(git rev-parse HEAD)"' in workflow
    assert 'os.environ["TAG_SHA"]' in workflow
    assert "${{ github.sha }}" not in workflow
