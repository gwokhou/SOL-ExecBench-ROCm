from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from sol_execbench.core.scoring.release_baseline import (
    ReleaseBaselineBundle,
    ReleaseBaselineVerification,
    ReleaseBaselineVerificationWorkload,
    ReleaseBaselineWorkload,
    ReleaseProvenance,
    write_release_baseline_bundle,
    write_release_baseline_verification,
)
from sol_execbench.core.integrity.checksums import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts/internal/release/check_prerelease_readiness.py"
SPEC = spec_from_file_location("check_prerelease_readiness", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_prerelease_readiness = module_from_spec(SPEC)
sys.modules[SPEC.name] = check_prerelease_readiness
SPEC.loader.exec_module(check_prerelease_readiness)


def _sha256(path: Path) -> str:
    return check_prerelease_readiness._sha256(path)


def _write_bundle(bundle_dir: Path, *, mutate: dict | None = None) -> dict:
    (bundle_dir / "release_candidate_validation").mkdir(parents=True)
    (bundle_dir / "transcripts").mkdir(parents=True)
    (
        bundle_dir / "release_candidate_validation/release_candidate_validation.json"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )
    (bundle_dir / "transcripts/release_candidate_validation.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "sol_execbench.prerelease_artifact_bundle.v1",
        "bundle_version": "v1.26-test",
        "claim_boundary": {
            "engineering_prerelease_only": True,
            "full_235_problem_validation": False,
            "upstream_solar_parity": False,
            "leaderboard_ready": False,
            "hard_sandbox": False,
            "native_host_validation_from_docker": False,
            "mi300x_cdna3_full_suite_validated": False,
            "cdna4_validated": False,
        },
        "known_gaps": [
            {
                "id": "mi300x_cdna3_full_suite",
                "status": "deferred",
                "description": "MI300X validation remains deferred.",
            },
            {
                "id": "cdna4_validation",
                "status": "unavailable",
                "description": "CDNA4 validation is unavailable.",
            },
        ],
        "authority_map": [
            {"id": "trace_jsonl", "authority_class": "canonical", "status": "deferred"},
            {
                "id": "release_validation",
                "authority_class": "diagnostic-only",
                "status": "present",
            },
            {
                "id": "bounded_dataset_slice",
                "authority_class": "provisional",
                "status": "deferred",
            },
            {
                "id": "paper_validation",
                "authority_class": "deferred",
                "status": "deferred",
            },
            {"id": "cdna4", "authority_class": "unavailable", "status": "unavailable"},
        ],
        "artifacts": [
            {
                "id": "release_candidate_validation_json",
                "path": "release_candidate_validation/release_candidate_validation.json",
                "authority_class": "diagnostic-only",
                "status": "present",
                "required": True,
            },
            {
                "id": "transcript_release_candidate_validation",
                "path": "transcripts/release_candidate_validation.json",
                "authority_class": "diagnostic-only",
                "status": "present",
                "required": False,
            },
        ],
    }
    for artifact in manifest["artifacts"]:
        artifact_path = artifact["path"]
        assert isinstance(artifact_path, str)
        path = bundle_dir / artifact_path
        artifact["sha256"] = _sha256(path)
    if mutate:
        _deep_update(manifest, mutate)
    manifest_path = bundle_dir / "prerelease_artifact_bundle.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{_sha256(path)}  {path.relative_to(bundle_dir).as_posix()}")
    (bundle_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _deep_update(target: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _load_report(output_dir: Path) -> dict:
    return json.loads(
        (output_dir / "prerelease_readiness.json").read_text(encoding="utf-8")
    )


def test_readiness_passes_for_complete_bundle(tmp_path):
    bundle_dir = tmp_path / "bundle"
    output_dir = tmp_path / "readiness"
    _write_bundle(bundle_dir)

    assert (
        check_prerelease_readiness.main(
            [
                "--bundle-dir",
                str(bundle_dir),
                "--output-dir",
                str(output_dir),
                "--skip-doc-claim-checks",
            ]
        )
        == 0
    )

    report = _load_report(output_dir)
    assert report["schema_version"] == "sol_execbench.prerelease_readiness.v1"
    assert report["overall_status"] == "passed"
    assert report["summary"]["known_gaps"]["deferred"] == 1
    assert report["summary"]["known_gaps"]["unavailable"] == 1
    assert (output_dir / "prerelease_readiness.md").exists()


def test_readiness_rejects_release_verification_workload_drift(tmp_path):
    bundle_dir = tmp_path / "bundle"
    manifest = _write_bundle(bundle_dir)
    release_dir = bundle_dir / "release_baseline"
    release_dir.mkdir()
    compact = release_dir / "scoring_baseline.json"
    compact.write_text('{"entries": []}\n', encoding="utf-8")
    baseline_path = release_dir / "release_baseline_bundle.json"
    baseline = ReleaseBaselineBundle(
        release="v2.14",
        suite_manifest_ref="suite.json",
        suite_manifest_sha256="a" * 64,
        baseline_artifact_ref=str(compact),
        baseline_artifact_sha256=sha256_file(compact),
        provenance=ReleaseProvenance(solution="solution", solution_sha256="b" * 64),
        workloads=(
            ReleaseBaselineWorkload(
                "gemm", "w1", "derived", 1.0, ("missing_bound_evidence",)
            ),
        ),
        latency_tolerance_rel=0.05,
    )
    write_release_baseline_bundle(baseline, baseline_path)
    verification_path = release_dir / "release_baseline_verification.json"
    write_release_baseline_verification(
        ReleaseBaselineVerification(
            release=baseline.release,
            bundle_ref=str(baseline_path),
            bundle_sha256=sha256_file(baseline_path),
            rerun_trace_ref="rerun.jsonl",
            rerun_trace_sha256="c" * 64,
            workloads=(
                ReleaseBaselineVerificationWorkload(
                    "gemm", "w1", "official", "official", 1.0, 1.0, 0.0, True, ()
                ),
            ),
        ),
        verification_path,
    )
    manifest["artifacts"].extend(
        [
            {
                "id": "release_scoring_baseline",
                "path": "release_baseline/scoring_baseline.json",
                "authority_class": "provisional",
                "status": "present",
                "required": True,
                "sha256": sha256_file(compact),
            },
            {
                "id": "release_baseline_bundle",
                "path": "release_baseline/release_baseline_bundle.json",
                "authority_class": "provisional",
                "status": "present",
                "required": True,
                "sha256": sha256_file(baseline_path),
            },
            {
                "id": "release_baseline_verification",
                "path": "release_baseline/release_baseline_verification.json",
                "authority_class": "provisional",
                "status": "present",
                "required": True,
                "sha256": sha256_file(verification_path),
            },
        ]
    )
    manifest["release_baseline_summary"] = baseline.summary
    (bundle_dir / "prerelease_artifact_bundle.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (bundle_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(bundle_dir)}"
            for path in sorted(bundle_dir.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        check_prerelease_readiness.main(
            [
                "--bundle-dir",
                str(bundle_dir),
                "--output-dir",
                str(tmp_path / "readiness"),
                "--skip-doc-claim-checks",
            ]
        )
        == 1
    )
    assert "release_baseline_workload_mismatch" in {
        finding["id"] for finding in _load_report(tmp_path / "readiness")["findings"]
    }


def test_readiness_fails_when_required_artifact_missing(tmp_path):
    bundle_dir = tmp_path / "bundle"
    output_dir = tmp_path / "readiness"
    _write_bundle(bundle_dir)
    (
        bundle_dir / "release_candidate_validation/release_candidate_validation.json"
    ).unlink()

    assert (
        check_prerelease_readiness.main(
            [
                "--bundle-dir",
                str(bundle_dir),
                "--output-dir",
                str(output_dir),
                "--skip-doc-claim-checks",
            ]
        )
        == 1
    )

    report = _load_report(output_dir)
    ids = {finding["id"] for finding in report["findings"]}
    assert "missing_artifact_file_release_candidate_validation_json" in ids
    assert report["summary"]["blocking"] >= 1


def test_readiness_rejects_a_lone_compact_release_baseline(tmp_path):
    bundle_dir = tmp_path / "bundle"
    manifest = _write_bundle(bundle_dir)
    compact_path = bundle_dir / "release_baseline/scoring_baseline.json"
    compact_path.parent.mkdir()
    compact_path.write_text('{"entries": []}\n', encoding="utf-8")
    manifest["artifacts"].append(
        {
            "id": "release_scoring_baseline",
            "path": "release_baseline/scoring_baseline.json",
            "authority_class": "provisional",
            "status": "present",
            "required": True,
            "sha256": sha256_file(compact_path),
        }
    )
    (bundle_dir / "prerelease_artifact_bundle.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (bundle_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(bundle_dir)}"
            for path in sorted(bundle_dir.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        check_prerelease_readiness.main(
            [
                "--bundle-dir",
                str(bundle_dir),
                "--output-dir",
                str(tmp_path / "readiness"),
                "--skip-doc-claim-checks",
            ]
        )
        == 1
    )
    assert "release_baseline_evidence_pair_missing" in {
        finding["id"] for finding in _load_report(tmp_path / "readiness")["findings"]
    }


def test_readiness_fails_on_claim_boundary_regression(tmp_path):
    bundle_dir = tmp_path / "bundle"
    output_dir = tmp_path / "readiness"
    _write_bundle(
        bundle_dir,
        mutate={"claim_boundary": {"leaderboard_ready": True}},
    )

    assert (
        check_prerelease_readiness.main(
            [
                "--bundle-dir",
                str(bundle_dir),
                "--output-dir",
                str(output_dir),
                "--skip-doc-claim-checks",
            ]
        )
        == 1
    )

    report = _load_report(output_dir)
    assert "forbidden_claim_leaderboard_ready" in {
        finding["id"] for finding in report["findings"]
    }


def test_readiness_fails_on_invalid_known_gap_status(tmp_path):
    bundle_dir = tmp_path / "bundle"
    output_dir = tmp_path / "readiness"
    _write_bundle(
        bundle_dir,
        mutate={
            "known_gaps": [
                {
                    "id": "paper_validation",
                    "status": "unreviewed",
                    "description": "not reviewed",
                }
            ]
        },
    )

    assert (
        check_prerelease_readiness.main(
            [
                "--bundle-dir",
                str(bundle_dir),
                "--output-dir",
                str(output_dir),
                "--skip-doc-claim-checks",
            ]
        )
        == 1
    )

    report = _load_report(output_dir)
    assert "invalid_known_gap_status_paper_validation" in {
        finding["id"] for finding in report["findings"]
    }


def test_readiness_doc_claim_checks_cover_public_boundaries(tmp_path):
    bundle_dir = tmp_path / "bundle"
    output_dir = tmp_path / "readiness"
    _write_bundle(bundle_dir)

    assert (
        check_prerelease_readiness.main(
            ["--bundle-dir", str(bundle_dir), "--output-dir", str(output_dir)]
        )
        == 0
    )


def test_provenance_policy_check_passes_for_repository_state():
    findings = check_prerelease_readiness._check_provenance_policy(REPO_ROOT)

    assert [finding for finding in findings if finding.status == "blocking"] == []


def test_provenance_policy_check_blocks_missing_manifest(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/user/provenance.md").write_text(
        "upstream retained derivative modified independent ROCm work not legal advice "
        "not imply NVIDIA or AMD endorsement\n",
        encoding="utf-8",
    )

    findings = check_prerelease_readiness._check_provenance_policy(tmp_path)

    assert any(finding.id == "missing_provenance_manifest" for finding in findings)


def test_provenance_policy_check_blocks_cleanup_candidate_with_nvidia_header(tmp_path):
    source_path = tmp_path / "src/package/new_rocm.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "\n".join(
            (
                check_prerelease_readiness.NVIDIA_HEADER,
                "# SPDX-License-Identifier: Apache-2.0",
                "",
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/user/provenance.md").write_text(
        "upstream retained derivative modified independent ROCm work not legal advice "
        "not imply NVIDIA or AMD endorsement\n",
        encoding="utf-8",
    )
    (tmp_path / "provenance.toml").write_text(
        "\n".join(
            (
                "[nvidia_notice]",
                "allowed = []",
                'cleanup_candidates = ["src/package/new_rocm.py"]',
                "",
            )
        ),
        encoding="utf-8",
    )

    findings = check_prerelease_readiness._check_provenance_policy(tmp_path)

    assert {finding.id for finding in findings} >= {
        "unexpected_nvidia_notice",
        "cleanup_candidate_header_mismatch",
    }


def test_readiness_blocks_restricted_nvidia_dataset_payload_in_bundle(tmp_path):
    bundle_dir = tmp_path / "bundle"
    output_dir = tmp_path / "readiness"
    _write_bundle(bundle_dir)
    restricted_path = (
        bundle_dir / "data/SOL-ExecBench/benchmark/L1/problem/definition.json"
    )
    restricted_path.parent.mkdir(parents=True)
    restricted_path.write_text("{}", encoding="utf-8")

    assert (
        check_prerelease_readiness.main(
            [
                "--bundle-dir",
                str(bundle_dir),
                "--output-dir",
                str(output_dir),
                "--skip-doc-claim-checks",
            ]
        )
        == 1
    )

    report = _load_report(output_dir)
    assert "restricted_dataset_in_release_bundle_nvidia_sol_execbench" in {
        finding["id"] for finding in report["findings"]
    }
