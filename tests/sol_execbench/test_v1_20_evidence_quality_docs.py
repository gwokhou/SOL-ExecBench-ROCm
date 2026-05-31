from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.claim_upgrade import ClaimUpgradeReport
from sol_execbench.core.consistency import ConsistencyReport
from sol_execbench.core.evaluation_stability import EvaluationStabilityReport
from sol_execbench.core.trust_summary import TrustSummaryReport

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE = REPO_ROOT / "docs/v1_20_evidence_quality_guide.md"
EXAMPLES_DIR = REPO_ROOT / "docs/examples/v1_20_evidence_quality"


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v1_20_guide_is_linked_from_public_entry_docs():
    for doc in ("docs/CLAIMS.md", "docs/RESEARCHER-GUIDE.md", "docs/TESTING.md"):
        assert "docs/v1_20_evidence_quality_guide.md" in _read(doc)


def test_v1_20_guide_names_scripts_and_boundaries():
    text = GUIDE.read_text(encoding="utf-8")

    for expected in (
        "scripts/report_consistency.py",
        "scripts/report_evaluation_stability.py",
        "scripts/report_claim_upgrade.py",
        "scripts/report_trust_summary.py",
        "sol_execbench.consistency_report.v1",
        "sol_execbench.evaluation_stability.v1",
        "sol_execbench.claim_upgrade.v1",
        "sol_execbench.trust_summary.v1",
        "not add full 235-problem paper validation",
        "CDNA3/MI300X/CDNA4 validation",
        "native-host Matrix authority",
        "hosted leaderboard readiness",
        "upstream SOLAR parity",
    ):
        assert expected in text


def test_v1_20_consistency_example_includes_amd_sol_and_solar_refs():
    text = GUIDE.read_text(encoding="utf-8")
    consistency_block = text.split("scripts/report_consistency.py", maxsplit=1)[1].split(
        "UV_CACHE_DIR=out/v1_20_demo/uv-cache uv run scripts/report_evaluation_stability.py",
        maxsplit=1,
    )[0]

    assert "--amd-sol-report out/v1_20_demo/amd_sol.json" in consistency_block
    assert "--solar-derivation out/v1_20_demo/solar_derivation.json" in consistency_block


def test_v1_20_example_readme_references_existing_fixture_files():
    readme = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")
    assert "../../v1_20_evidence_quality_guide.md" in readme

    for fixture in (
        "trust_summary.consistent.demo.json",
        "consistency.contradictory.demo.json",
        "evaluation_stability.noisy.demo.json",
        "claim_upgrade.blocked.demo.json",
        "trust_summary.missing.demo.json",
    ):
        assert (EXAMPLES_DIR / fixture).exists()


def test_v1_20_example_json_fixtures_validate_against_real_models():
    ConsistencyReport.model_validate(
        _json(EXAMPLES_DIR / "consistency.contradictory.demo.json")
    )
    EvaluationStabilityReport.model_validate(
        _json(EXAMPLES_DIR / "evaluation_stability.noisy.demo.json")
    )
    ClaimUpgradeReport.model_validate(
        _json(EXAMPLES_DIR / "claim_upgrade.blocked.demo.json")
    )
    TrustSummaryReport.model_validate(
        _json(EXAMPLES_DIR / "trust_summary.missing.demo.json")
    )
    TrustSummaryReport.model_validate(
        _json(EXAMPLES_DIR / "trust_summary.consistent.demo.json")
    )


def test_v1_20_example_refs_are_relative_bounded_and_checksum_backed():
    for path in EXAMPLES_DIR.glob("*.json"):
        payload = _json(path)
        text = json.dumps(payload, sort_keys=True)
        assert "/tmp/" not in text
        assert "/home/" not in text
        assert "out/v1_20_demo/" in text
        assert "sha256:" in text


def test_v1_20_example_authority_flags_remain_false_or_diagnostic_only():
    for path in EXAMPLES_DIR.glob("*.json"):
        payload = _json(path)
        text = json.dumps(payload, sort_keys=True)
        for forbidden_truth in (
            '"score_authority": true',
            '"paper_parity": true',
            '"leaderboard_authority": true',
            '"native_host_validation": true',
            '"new_hardware_validation": true',
            '"paper_validation": true',
        ):
            assert forbidden_truth not in text


def test_v1_20_docs_keep_required_negative_claim_boundaries_visible():
    combined = "\n".join(
        [
            GUIDE.read_text(encoding="utf-8"),
            _read("docs/CLAIMS.md"),
            _read("docs/RESEARCHER-GUIDE.md"),
            _read("docs/TESTING.md"),
            (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8"),
        ]
    )

    for expected in (
        "not paper validation",
        "not paper parity",
        "not leaderboard authority",
        "not native-host validation",
        "not new-hardware validation",
        "does not create correctness",
        "score authority",
    ):
        assert expected in combined
