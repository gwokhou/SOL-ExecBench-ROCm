#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Check prerelease artifact bundle readiness before publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Sequence

from sol_execbench.core.scoring.release_baseline import (
    load_release_baseline_bundle,
    release_baseline_verification_from_dict,
)

SCHEMA_VERSION = "sol_execbench.prerelease_readiness.v1"
BUNDLE_SCHEMA_VERSION = "sol_execbench.prerelease_artifact_bundle.v1"
DEFAULT_OUTPUT_DIR = Path("out/prerelease_readiness")
REPO_ROOT = Path(__file__).resolve().parents[3]
PROVENANCE_MANIFEST_PATH = Path("provenance.toml")
PROVENANCE_DOC_PATH = Path("docs/provenance.md")
DATASET_REDISTRIBUTION_SCRIPT_PATH = Path("scripts/check_dataset_redistribution.py")
NVIDIA_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved."
)
PROJECT_HEADER = "# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port"
AUTHORITY_CLASSES = {
    "canonical",
    "diagnostic-only",
    "provisional",
    "deferred",
    "unavailable",
}
KNOWN_GAP_STATUSES = {
    "blocking",
    "deferred",
    "unavailable",
    "diagnostic-only",
}
FORBIDDEN_TRUTHY_CLAIMS = {
    "full_235_problem_validation",
    "upstream_solar_parity",
    "leaderboard_ready",
    "hard_sandbox",
    "native_host_validation_from_docker",
    "mi300x_cdna3_full_suite_validated",
    "cdna4_validated",
}
REQUIRED_DOC_PHRASES = {
    "docs/CLAIMS.md": (
        "MI300X and MI308X are sibling GPU products",
        "recorded on MI308X",
        "CDNA4 validation is unavailable",
        "not native-host validation",
    ),
    "docs/prerelease_artifact_bundle.md": (
        "engineering prerelease and research preview evidence only",
        "MI300X and MI308X are sibling GPU products",
        "recorded on MI308X",
        "CDNA4 validation is unavailable",
    ),
}
REQUIRED_PROVENANCE_DOC_PHRASES = (
    "upstream retained",
    "derivative modified",
    "independent ROCm work",
    "not legal advice",
    "not imply NVIDIA or AMD endorsement",
)


@dataclass(frozen=True)
class Finding:
    id: str
    status: str
    category: str
    message: str
    path: str | None = None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = args.bundle_dir
    findings: list[Finding] = []

    manifest_path = bundle_dir / "prerelease_artifact_bundle.json"
    checksum_path = bundle_dir / "SHA256SUMS"
    manifest = _load_json(manifest_path, findings)
    checksums = _load_checksums(checksum_path, findings)
    checksum_cache: dict[Path, str] = {}
    findings.extend(_check_provenance_policy())
    if manifest:
        findings.extend(
            _check_manifest(manifest, bundle_dir, checksums, checksum_cache)
        )
        findings.extend(_check_dataset_release_redistribution(bundle_dir))
        if not args.skip_doc_claim_checks:
            findings.extend(_check_doc_claims())

    payload = _build_payload(bundle_dir, findings, manifest)
    json_path = args.output_dir / "prerelease_readiness.json"
    markdown_path = args.output_dir / "prerelease_readiness.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return 1 if payload["overall_status"] == "blocking" else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check prerelease artifact bundle readiness before publication.",
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-doc-claim-checks", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path, findings: list[Finding]) -> dict[str, object] | None:
    if not path.exists():
        findings.append(
            Finding(
                id="missing_manifest",
                status="blocking",
                category="artifact",
                message="Missing prerelease_artifact_bundle.json.",
                path=path.as_posix(),
            )
        )
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        findings.append(
            Finding(
                id="invalid_manifest_json",
                status="blocking",
                category="artifact",
                message=f"Manifest is not valid JSON: {exc}",
                path=path.as_posix(),
            )
        )
        return None
    if not isinstance(payload, dict):
        findings.append(
            Finding(
                id="invalid_manifest_shape",
                status="blocking",
                category="artifact",
                message="Manifest root must be a JSON object.",
                path=path.as_posix(),
            )
        )
        return None
    return payload


def _load_checksums(path: Path, findings: list[Finding]) -> dict[str, str]:
    if not path.exists():
        findings.append(
            Finding(
                id="missing_checksums",
                status="blocking",
                category="artifact",
                message="Missing SHA256SUMS.",
                path=path.as_posix(),
            )
        )
        return {}
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            digest, rel_path = line.split(maxsplit=1)
        except ValueError:
            findings.append(
                Finding(
                    id="malformed_checksum_line",
                    status="blocking",
                    category="checksum",
                    message=f"Malformed checksum line: {line}",
                    path=path.as_posix(),
                )
            )
            continue
        checksums[rel_path.strip()] = digest.strip()
    return checksums


def _check_manifest(
    manifest: dict[str, object],
    bundle_dir: Path,
    checksums: dict[str, str],
    checksum_cache: dict[Path, str],
) -> list[Finding]:
    findings: list[Finding] = []
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        findings.append(
            Finding(
                id="manifest_schema_mismatch",
                status="blocking",
                category="artifact",
                message=f"Expected schema {BUNDLE_SCHEMA_VERSION}.",
            )
        )
    findings.extend(_check_authority_classes(manifest))
    findings.extend(_check_claim_boundary(manifest))
    findings.extend(_check_known_gaps(manifest))
    findings.extend(_check_artifacts(manifest, bundle_dir, checksum_cache))
    findings.extend(_check_release_baseline_evidence(manifest, bundle_dir))
    findings.extend(_check_bundle_checksums(bundle_dir, checksums, checksum_cache))
    return findings


def _check_release_baseline_evidence(
    manifest: dict[str, object], bundle_dir: Path
) -> list[Finding]:
    """Validate the linked release baseline and independent rerun evidence."""
    artifacts = {
        str(item.get("id")): item for item in _list_of_dicts(manifest.get("artifacts"))
    }
    baseline_artifact = artifacts.get("release_baseline_bundle")
    verification_artifact = artifacts.get("release_baseline_verification")
    scoring_artifact = artifacts.get("release_scoring_baseline")
    if baseline_artifact is None and verification_artifact is None:
        return []
    if (
        baseline_artifact is None
        or verification_artifact is None
        or scoring_artifact is None
    ):
        return [
            Finding(
                id="release_baseline_evidence_pair_missing",
                status="blocking",
                category="release_baseline",
                message="Release baseline bundle and verification must both be present.",
            )
        ]
    paths = []
    for artifact in (baseline_artifact, verification_artifact, scoring_artifact):
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            return [
                Finding(
                    id="release_baseline_evidence_path_missing",
                    status="blocking",
                    category="release_baseline",
                    message="Release baseline evidence path is missing.",
                )
            ]
        paths.append(_resolve_artifact_path(bundle_dir, Path(path)))
    baseline_path, verification_path, scoring_path = paths
    try:
        baseline = load_release_baseline_bundle(baseline_path)
        verification = release_baseline_verification_from_dict(
            json.loads(verification_path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [
            Finding(
                id="invalid_release_baseline_evidence",
                status="blocking",
                category="release_baseline",
                message=str(exc),
            )
        ]
    findings: list[Finding] = []
    if verification.bundle_sha256 != _sha256(baseline_path):
        findings.append(
            Finding(
                id="release_baseline_verification_checksum_mismatch",
                status="blocking",
                category="release_baseline",
                message="Verification does not reference the bundled baseline digest.",
            )
        )
    if _sha256(scoring_path) != baseline.baseline_artifact_sha256:
        findings.append(
            Finding(
                id="release_scoring_baseline_checksum_mismatch",
                status="blocking",
                category="release_baseline",
                message="Bundled compact scoring baseline does not match release evidence.",
            )
        )
    baseline_summary = baseline.summary
    verification_summary = verification.summary
    if any(
        verification_summary[key] != baseline_summary[key] for key in baseline_summary
    ):
        findings.append(
            Finding(
                id="release_baseline_summary_mismatch",
                status="blocking",
                category="release_baseline",
                message="Verification classifications do not preserve the bundle denominator.",
            )
        )
    baseline_by_key = {row.key: row for row in baseline.workloads}
    verification_by_key = {row.key: row for row in verification.workloads}
    if set(verification_by_key) != set(baseline_by_key) or any(
        verification_row.original_classification != baseline_row.classification
        or verification_row.classification != baseline_row.classification
        or verification_row.passed != (baseline_row.classification != "blocked")
        for key, baseline_row in baseline_by_key.items()
        if (verification_row := verification_by_key.get(key)) is not None
    ):
        findings.append(
            Finding(
                id="release_baseline_workload_mismatch",
                status="blocking",
                category="release_baseline",
                message="Verification workload identities and outcomes must exactly preserve the baseline evidence.",
            )
        )
    manifest_summary = manifest.get("release_baseline_summary")
    if manifest_summary != baseline_summary:
        findings.append(
            Finding(
                id="release_baseline_manifest_summary_mismatch",
                status="blocking",
                category="release_baseline",
                message="Manifest release-baseline summary does not match evidence.",
            )
        )
    boundary = manifest.get("claim_boundary")
    if (
        isinstance(boundary, dict)
        and baseline_summary["derived"] + baseline_summary["blocked"]
    ):
        if boundary.get("release_baseline_full_suite_official") is not False:
            findings.append(
                Finding(
                    id="release_baseline_full_suite_authority_overclaim",
                    status="blocking",
                    category="claim_boundary",
                    message="Derived or blocked release-baseline rows forbid a full-suite official claim.",
                )
            )
    return findings


def _check_authority_classes(manifest: dict[str, object]) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[str] = set()
    for section in ("artifacts", "authority_map"):
        for entry in _list_of_dicts(manifest.get(section)):
            authority_class = str(entry.get("authority_class", ""))
            seen.add(authority_class)
            if authority_class not in AUTHORITY_CLASSES:
                findings.append(
                    Finding(
                        id="unknown_authority_class",
                        status="blocking",
                        category="authority",
                        message=f"Unknown authority class {authority_class!r}.",
                        path=str(entry.get("path") or entry.get("id") or section),
                    )
                )
    missing = AUTHORITY_CLASSES - seen
    if missing:
        findings.append(
            Finding(
                id="missing_authority_class",
                status="blocking",
                category="authority",
                message=f"Manifest does not cover authority classes: {', '.join(sorted(missing))}.",
            )
        )
    return findings


def _check_claim_boundary(manifest: dict[str, object]) -> list[Finding]:
    findings: list[Finding] = []
    claim_boundary = manifest.get("claim_boundary")
    if not isinstance(claim_boundary, dict):
        return [
            Finding(
                id="missing_claim_boundary",
                status="blocking",
                category="claim_boundary",
                message="Manifest is missing claim_boundary object.",
            )
        ]
    for key in sorted(FORBIDDEN_TRUTHY_CLAIMS):
        if claim_boundary.get(key) is not False:
            findings.append(
                Finding(
                    id=f"forbidden_claim_{key}",
                    status="blocking",
                    category="claim_boundary",
                    message=f"Forbidden claim boundary must be false: {key}.",
                )
            )
    if claim_boundary.get("engineering_prerelease_only") is not True:
        findings.append(
            Finding(
                id="missing_engineering_prerelease_boundary",
                status="blocking",
                category="claim_boundary",
                message="engineering_prerelease_only must be true.",
            )
        )
    return findings


def _check_known_gaps(manifest: dict[str, object]) -> list[Finding]:
    findings: list[Finding] = []
    known_gaps = _list_of_dicts(manifest.get("known_gaps"))
    if not known_gaps:
        findings.append(
            Finding(
                id="missing_known_gaps",
                status="blocking",
                category="known_gap",
                message="Manifest must report known gaps before publishing.",
            )
        )
        return findings
    for gap in known_gaps:
        status = str(gap.get("status", ""))
        gap_id = str(gap.get("id", "unknown_gap"))
        if status not in KNOWN_GAP_STATUSES:
            findings.append(
                Finding(
                    id=f"invalid_known_gap_status_{gap_id}",
                    status="blocking",
                    category="known_gap",
                    message=f"Known gap {gap_id} has invalid status {status!r}.",
                )
            )
        else:
            findings.append(
                Finding(
                    id=f"known_gap_{gap_id}",
                    status=status,
                    category="known_gap",
                    message=str(gap.get("description", "")),
                )
            )
    return findings


def _check_artifacts(
    manifest: dict[str, object],
    bundle_dir: Path,
    checksum_cache: dict[Path, str],
) -> list[Finding]:
    findings: list[Finding] = []
    for artifact in _list_of_dicts(manifest.get("artifacts")):
        artifact_id = str(artifact.get("id", "unknown_artifact"))
        required = artifact.get("required") is True
        status = str(artifact.get("status", ""))
        path_value = artifact.get("path")
        if required and status != "present":
            findings.append(
                Finding(
                    id=f"required_artifact_not_present_{artifact_id}",
                    status="blocking",
                    category="artifact",
                    message=f"Required artifact {artifact_id} is not present.",
                    path=str(path_value) if path_value else None,
                )
            )
        if not isinstance(path_value, str) or not path_value:
            continue
        path = _resolve_artifact_path(bundle_dir, Path(path_value))
        if status == "present" and not path.exists():
            findings.append(
                Finding(
                    id=f"missing_artifact_file_{artifact_id}",
                    status="blocking",
                    category="artifact",
                    message=f"Artifact file is missing: {path_value}.",
                    path=path_value,
                )
            )
            continue
        expected_sha = artifact.get("sha256")
        if status == "present" and isinstance(expected_sha, str) and path.exists():
            actual_sha = _sha256_cached(path, checksum_cache)
            if actual_sha != expected_sha:
                findings.append(
                    Finding(
                        id=f"artifact_checksum_mismatch_{artifact_id}",
                        status="blocking",
                        category="checksum",
                        message=f"Artifact checksum mismatch for {artifact_id}.",
                        path=path_value,
                    )
                )
    return findings


def _check_bundle_checksums(
    bundle_dir: Path,
    checksums: dict[str, str],
    checksum_cache: dict[Path, str],
) -> list[Finding]:
    findings: list[Finding] = []
    files: list[Path] = []
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        files.append(path)
    _sha256_cached_many(files, checksum_cache)
    for path in files:
        rel_path = path.relative_to(bundle_dir).as_posix()
        expected = checksums.get(rel_path)
        if expected is None:
            findings.append(
                Finding(
                    id="missing_checksum_entry",
                    status="blocking",
                    category="checksum",
                    message=f"Missing checksum entry for {rel_path}.",
                    path=rel_path,
                )
            )
            continue
        actual = _sha256_cached(path, checksum_cache)
        if actual != expected:
            findings.append(
                Finding(
                    id="checksum_mismatch",
                    status="blocking",
                    category="checksum",
                    message=f"Checksum mismatch for {rel_path}.",
                    path=rel_path,
                )
            )
    return findings


def _check_doc_claims() -> list[Finding]:
    findings: list[Finding] = []
    for path_text, phrases in REQUIRED_DOC_PHRASES.items():
        path = Path(path_text)
        if not path.exists():
            findings.append(
                Finding(
                    id="missing_claim_doc",
                    status="blocking",
                    category="claim_boundary",
                    message=f"Missing claim-boundary doc: {path_text}.",
                    path=path_text,
                )
            )
            continue
        text = path.read_text(encoding="utf-8")
        for phrase in phrases:
            if phrase not in text:
                findings.append(
                    Finding(
                        id="missing_claim_phrase",
                        status="blocking",
                        category="claim_boundary",
                        message=f"Missing claim-boundary phrase: {phrase}",
                        path=path_text,
                    )
                )
    return findings


def _check_provenance_policy(root: Path = REPO_ROOT) -> list[Finding]:
    findings: list[Finding] = []
    manifest_path = root / PROVENANCE_MANIFEST_PATH
    doc_path = root / PROVENANCE_DOC_PATH

    if not manifest_path.exists():
        return [
            Finding(
                id="missing_provenance_manifest",
                status="blocking",
                category="provenance",
                message="Missing provenance.toml.",
                path=PROVENANCE_MANIFEST_PATH.as_posix(),
            )
        ]
    try:
        provenance = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return [
            Finding(
                id="invalid_provenance_manifest",
                status="blocking",
                category="provenance",
                message=f"provenance.toml is not valid TOML: {exc}",
                path=PROVENANCE_MANIFEST_PATH.as_posix(),
            )
        ]

    if not doc_path.exists():
        findings.append(
            Finding(
                id="missing_provenance_doc",
                status="blocking",
                category="provenance",
                message="Missing docs/provenance.md.",
                path=PROVENANCE_DOC_PATH.as_posix(),
            )
        )
    else:
        doc_text = doc_path.read_text(encoding="utf-8").lower()
        for phrase in REQUIRED_PROVENANCE_DOC_PHRASES:
            if phrase.lower() not in doc_text:
                findings.append(
                    Finding(
                        id="missing_provenance_doc_phrase",
                        status="blocking",
                        category="provenance",
                        message=f"Missing provenance doc phrase: {phrase}",
                        path=PROVENANCE_DOC_PATH.as_posix(),
                    )
                )

    nvidia_notice = provenance.get("nvidia_notice")
    dataset_policy = provenance.get("dataset_policy")
    if not isinstance(dataset_policy, dict):
        findings.append(
            Finding(
                id="missing_dataset_policy",
                status="blocking",
                category="provenance",
                message="provenance.toml must define [dataset_policy].",
                path=PROVENANCE_MANIFEST_PATH.as_posix(),
            )
        )
    else:
        findings.extend(_check_dataset_policy(dataset_policy))

    if not isinstance(nvidia_notice, dict):
        findings.append(
            Finding(
                id="missing_provenance_nvidia_notice_section",
                status="blocking",
                category="provenance",
                message="provenance.toml must define [nvidia_notice].",
                path=PROVENANCE_MANIFEST_PATH.as_posix(),
            )
        )
        return findings

    allowed = _string_set(nvidia_notice.get("allowed"))
    cleanup_candidates = _string_set(nvidia_notice.get("cleanup_candidates"))
    if allowed & cleanup_candidates:
        findings.append(
            Finding(
                id="provenance_lists_overlap",
                status="blocking",
                category="provenance",
                message="nvidia_notice.allowed and cleanup_candidates must not overlap.",
                path=PROVENANCE_MANIFEST_PATH.as_posix(),
            )
        )

    active_nvidia_headers = _active_nvidia_header_files(root)
    unexpected = active_nvidia_headers - allowed
    missing = allowed - active_nvidia_headers
    if unexpected:
        findings.append(
            Finding(
                id="unexpected_nvidia_notice",
                status="blocking",
                category="provenance",
                message=f"Unexpected NVIDIA notices: {', '.join(sorted(unexpected))}.",
                path=PROVENANCE_MANIFEST_PATH.as_posix(),
            )
        )
    if missing:
        findings.append(
            Finding(
                id="missing_allowed_nvidia_notice",
                status="blocking",
                category="provenance",
                message=f"Allowed files missing NVIDIA notices: {', '.join(sorted(missing))}.",
                path=PROVENANCE_MANIFEST_PATH.as_posix(),
            )
        )

    for relative_path in sorted(allowed):
        path = root / relative_path
        if not path.exists():
            findings.append(_missing_provenance_file(relative_path))
            continue
        lines = path.read_text(encoding="utf-8").splitlines()[:4]
        if NVIDIA_HEADER not in lines or PROJECT_HEADER not in lines:
            findings.append(
                Finding(
                    id="allowed_file_header_mismatch",
                    status="blocking",
                    category="provenance",
                    message="Allowed NVIDIA notice file must contain NVIDIA and project attribution.",
                    path=relative_path,
                )
            )

    for relative_path in sorted(cleanup_candidates):
        path = root / relative_path
        if not path.exists():
            findings.append(_missing_provenance_file(relative_path))
            continue
        lines = path.read_text(encoding="utf-8").splitlines()[:4]
        if NVIDIA_HEADER in lines or PROJECT_HEADER not in lines:
            findings.append(
                Finding(
                    id="cleanup_candidate_header_mismatch",
                    status="blocking",
                    category="provenance",
                    message="Cleanup candidate must contain project attribution and no NVIDIA file attribution.",
                    path=relative_path,
                )
            )

    return findings


def _check_dataset_policy(dataset_policy: dict[str, object]) -> list[Finding]:
    findings: list[Finding] = []
    if (
        dataset_policy.get("schema_version")
        != "sol_execbench.dataset_provenance_policy.v1"
    ):
        findings.append(
            Finding(
                id="dataset_policy_schema_mismatch",
                status="blocking",
                category="provenance",
                message="Dataset policy schema must be sol_execbench.dataset_provenance_policy.v1.",
                path=PROVENANCE_MANIFEST_PATH.as_posix(),
            )
        )
    sources = _list_of_dicts(dataset_policy.get("sources"))
    source_ids = {str(source.get("id", "")) for source in sources}
    required_sources = {
        "nvidia_sol_execbench",
        "flashinfer_trace",
        "generated_local_migration_artifacts",
        "project_owned_rocm_code",
    }
    missing = required_sources - source_ids
    if missing:
        findings.append(
            Finding(
                id="dataset_policy_missing_sources",
                status="blocking",
                category="provenance",
                message=f"Dataset policy missing sources: {', '.join(sorted(missing))}.",
                path=PROVENANCE_MANIFEST_PATH.as_posix(),
            )
        )
    nvidia_source = next(
        (source for source in sources if source.get("id") == "nvidia_sol_execbench"),
        None,
    )
    if nvidia_source is None:
        return findings
    if (
        nvidia_source.get("license") != "NVIDIA Evaluation Dataset License"
        or nvidia_source.get("repository_redistribution") is not False
        or nvidia_source.get("release_bundle_redistribution") is not False
    ):
        findings.append(
            Finding(
                id="nvidia_dataset_boundary_mismatch",
                status="blocking",
                category="provenance",
                message=(
                    "NVIDIA SOL-ExecBench dataset policy must preserve NVIDIA Evaluation "
                    "Dataset License and block repository/release redistribution."
                ),
                path=PROVENANCE_MANIFEST_PATH.as_posix(),
            )
        )
    return findings


def _check_dataset_release_redistribution(bundle_dir: Path) -> list[Finding]:
    checker = _load_dataset_redistribution_checker()
    policy = checker.load_dataset_policy(REPO_ROOT / PROVENANCE_MANIFEST_PATH)
    return [
        Finding(
            id=f"restricted_dataset_in_release_bundle_{finding.source_id}",
            status="blocking",
            category="provenance",
            message=finding.message,
            path=finding.path,
        )
        for finding in checker.check_release_root(bundle_dir, policy)
    ]


def _load_dataset_redistribution_checker():
    path = REPO_ROOT / DATASET_REDISTRIBUTION_SCRIPT_PATH
    spec = spec_from_file_location("check_dataset_redistribution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_payload(
    bundle_dir: Path,
    findings: list[Finding],
    manifest: dict[str, object] | None,
) -> dict[str, object]:
    blocking_count = sum(finding.status == "blocking" for finding in findings)
    known_gap_counts = {
        status: sum(
            finding.category == "known_gap" and finding.status == status
            for finding in findings
        )
        for status in sorted(KNOWN_GAP_STATUSES)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bundle_dir": bundle_dir.as_posix(),
        "bundle_version": manifest.get("bundle_version") if manifest else None,
        "overall_status": "blocking" if blocking_count else "passed",
        "summary": {
            "findings": len(findings),
            "blocking": blocking_count,
            "known_gaps": known_gap_counts,
        },
        "findings": [asdict(finding) for finding in findings],
    }


def _render_markdown(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, dict)
    lines = [
        "# Prerelease Readiness",
        "",
        f"**Schema:** `{payload['schema_version']}`",
        f"**Bundle:** `{payload['bundle_version']}`",
        f"**Overall status:** `{payload['overall_status']}`",
        "",
        "## Summary",
        "",
        f"- Findings: {summary['findings']}",
        f"- Blocking: {summary['blocking']}",
        "",
        "## Findings",
        "",
        "| Status | Category | ID | Path | Message |",
        "| --- | --- | --- | --- | --- |",
    ]
    for finding in payload["findings"]:
        assert isinstance(finding, dict)
        lines.append(
            f"| `{finding['status']}` | `{finding['category']}` | `{finding['id']}` | "
            f"`{finding.get('path') or ''}` | {finding['message']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def _active_nvidia_header_files(root: Path) -> set[str]:
    files: set[str] = set()
    for root_name in ("src", "scripts", "tests"):
        root_path = root / root_name
        if not root_path.exists():
            continue
        for path in root_path.rglob("*.py"):
            if NVIDIA_HEADER in path.read_text(encoding="utf-8").splitlines()[:4]:
                files.add(path.relative_to(root).as_posix())
    return files


def _missing_provenance_file(relative_path: str) -> Finding:
    return Finding(
        id="missing_provenance_file",
        status="blocking",
        category="provenance",
        message=f"Provenance manifest references a missing file: {relative_path}.",
        path=relative_path,
    )


def _resolve_artifact_path(bundle_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    in_bundle = bundle_dir / path
    if in_bundle.exists():
        return in_bundle
    return path


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _sha256_cached(path: Path, checksum_cache: dict[Path, str]) -> str:
    resolved = path.resolve()
    digest = checksum_cache.get(resolved)
    if digest is None:
        digest = _sha256(path)
        checksum_cache[resolved] = digest
    return digest


def _sha256_cached_many(paths: list[Path], checksum_cache: dict[Path, str]) -> None:
    missing = [path for path in paths if path.resolve() not in checksum_cache]
    if not missing:
        return
    workers = min(os.cpu_count() or 1, len(missing), 8)
    if workers <= 1:
        for path in missing:
            checksum_cache[path.resolve()] = _sha256(path)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for path, digest in zip(missing, executor.map(_sha256, missing), strict=True):
            checksum_cache[path.resolve()] = digest


if __name__ == "__main__":
    raise SystemExit(main())
