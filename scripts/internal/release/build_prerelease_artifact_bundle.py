#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Build a versioned prerelease artifact bundle for review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from sol_execbench.core.scoring.release_baseline import (
    load_release_baseline_bundle,
    release_baseline_verification_from_dict,
)

SCHEMA_VERSION = "sol_execbench.prerelease_artifact_bundle.v1"
DEFAULT_OUTPUT_DIR = Path("out/prerelease_artifact_bundle")
DEFAULT_TEMP_ROOT = Path("tmp/prerelease_artifact_bundle")
DEFAULT_LOG_TAIL_CHARS = 4000
AUTHORITY_CLASSES = {
    "canonical",
    "diagnostic-only",
    "provisional",
    "deferred",
    "unavailable",
}

DEFAULT_RELEASE_VALIDATION_COMMAND = [
    "uv",
    "run",
    "scripts/internal/release/release_candidate_validation.py",
    "--output-dir",
    "{release_validation_dir}",
]
DEFAULT_ENVIRONMENT_COMMAND = ["uv", "run", "sol-execbench", "doctor", "--json"]

TOKEN_PATTERN = re.compile(
    r"(?ix)"
    r"("
    r"(?:[A-Z0-9_]*?(?:TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|CREDENTIAL)[A-Z0-9_-]*?)"
    r"|authorization"
    r")"
    r"(\s*:\s*bearer\s+|\s*[:=]\s*)"
    r"([^\s'\"]+)"
)
_TOKEN_PREFIX_OVERLAP_CHARS = 512
_TOKEN_VALUE_DELIMITERS = set(" \t\r\n'\"")


@dataclass(frozen=True)
class CommandTranscript:
    name: str
    command: list[str]
    status: str
    classification: str
    duration_s: float
    transcript_path: str
    returncode: int | None = None
    next_action: str = "Review recorded artifacts before publishing."


@dataclass(frozen=True)
class BundleArtifact:
    id: str
    path: str | None
    authority_class: str
    status: str
    description: str
    sha256: str | None = None
    required: bool = False


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    bundle_dir = args.output_dir
    release_validation_dir = bundle_dir / "release_candidate_validation"
    transcript_dir = bundle_dir / "transcripts"
    environment_dir = bundle_dir / "environment"
    temp_dir = DEFAULT_TEMP_ROOT
    for path in (bundle_dir, release_validation_dir, transcript_dir, environment_dir):
        path.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    transcripts: list[CommandTranscript] = []
    artifacts: list[BundleArtifact] = []
    known_gaps = _default_known_gaps()
    checksum_cache: dict[Path, str] = {}

    release_baseline_summary: dict[str, int] | None = None
    if (args.release_baseline_bundle is None) != (
        args.release_baseline_verification is None
    ):
        raise SystemExit(
            "--release-baseline-bundle and --release-baseline-verification "
            "must be supplied together"
        )
    if args.release_baseline_bundle is not None:
        release_artifacts, release_baseline_summary = _release_baseline_artifacts(
            bundle_path=args.release_baseline_bundle,
            verification_path=args.release_baseline_verification,
            output_dir=bundle_dir,
            checksum_cache=checksum_cache,
        )
        artifacts.extend(release_artifacts)

    release_command = _expand_command(
        _command_from_args(
            args.release_validation_command,
            DEFAULT_RELEASE_VALIDATION_COMMAND,
        ),
        bundle_dir=bundle_dir,
        release_validation_dir=release_validation_dir,
        environment_dir=environment_dir,
    )
    if args.skip_release_validation:
        artifacts.append(
            BundleArtifact(
                id="release_candidate_validation",
                path=None,
                authority_class="diagnostic-only",
                status="deferred",
                description="Release-candidate validation was skipped by request.",
                required=True,
            )
        )
        known_gaps.append(
            {
                "id": "release_validation_skipped",
                "status": "deferred",
                "description": "Run without --skip-release-validation before publishing.",
            }
        )
    else:
        result = _run_command(
            name="release_candidate_validation",
            command=release_command,
            transcript_dir=transcript_dir,
            temp_dir=temp_dir,
            log_tail_chars=args.log_tail_chars,
            failure_classification="blocking",
            failure_next_action="Fix release-candidate validation before publishing.",
        )
        transcripts.append(result)
        artifacts.extend(
            _release_validation_artifacts(
                release_validation_dir,
                result.status,
                checksum_cache=checksum_cache,
            )
        )

    environment_command = _expand_command(
        _command_from_args(args.environment_command, DEFAULT_ENVIRONMENT_COMMAND),
        bundle_dir=bundle_dir,
        release_validation_dir=release_validation_dir,
        environment_dir=environment_dir,
    )
    if args.skip_environment_evidence:
        artifacts.append(
            BundleArtifact(
                id="environment_evidence",
                path=None,
                authority_class="diagnostic-only",
                status="deferred",
                description="Environment evidence command was skipped by request.",
            )
        )
    else:
        environment_result = _run_command(
            name="environment_evidence",
            command=environment_command,
            transcript_dir=transcript_dir,
            temp_dir=temp_dir,
            log_tail_chars=args.log_tail_chars,
            failure_classification="diagnostic-only",
            failure_next_action="Run on the release host and attach doctor output when available.",
            stdout_artifact=environment_dir / "sol_execbench_doctor.json",
        )
        transcripts.append(environment_result)
        doctor_path = environment_dir / "sol_execbench_doctor.json"
        artifacts.append(
            _file_artifact(
                id="environment_evidence",
                path=doctor_path,
                bundle_dir=bundle_dir,
                authority_class="diagnostic-only",
                status="present" if doctor_path.exists() else "unavailable",
                description=(
                    "Diagnostic environment evidence from sol-execbench doctor; "
                    "not timing, score, paper-parity, leaderboard, or hardware-validation authority."
                ),
                checksum_cache=checksum_cache,
            )
        )

    for transcript in transcripts:
        artifacts.append(
            _file_artifact(
                id=f"transcript_{transcript.name}",
                path=bundle_dir / transcript.transcript_path,
                bundle_dir=bundle_dir,
                authority_class="diagnostic-only",
                status="present",
                description=f"Command transcript for {transcript.name}.",
                checksum_cache=checksum_cache,
            )
        )

    artifacts.extend(
        _source_reference_artifacts(bundle_dir, checksum_cache=checksum_cache)
    )
    authority_map = _authority_map()
    _validate_authority_classes(artifacts, authority_map)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "bundle_version": args.version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": _git_context(args.source_ref),
        "overall_status": _overall_status(transcripts, artifacts),
        "summary": _summary(transcripts, artifacts),
        "claim_boundary": _claim_boundary(),
        "known_gaps": known_gaps,
        "authority_map": authority_map,
        "commands": [asdict(transcript) for transcript in transcripts],
        "artifacts": [asdict(artifact) for artifact in artifacts],
    }
    if release_baseline_summary is not None:
        payload["release_baseline_summary"] = release_baseline_summary

    manifest_path = bundle_dir / "prerelease_artifact_bundle.json"
    markdown_path = bundle_dir / "prerelease_artifact_bundle.md"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    _write_checksums(bundle_dir, checksum_cache)
    return 1 if payload["overall_status"] == "blocking" else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a versioned SOL ExecBench ROCm prerelease artifact bundle.",
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-ref", default=None)
    parser.add_argument("--skip-release-validation", action="store_true")
    parser.add_argument("--release-validation-command", nargs="+", default=None)
    parser.add_argument("--skip-environment-evidence", action="store_true")
    parser.add_argument("--environment-command", nargs="+", default=None)
    parser.add_argument("--release-baseline-bundle", type=Path, default=None)
    parser.add_argument("--release-baseline-verification", type=Path, default=None)
    parser.add_argument("--log-tail-chars", type=int, default=DEFAULT_LOG_TAIL_CHARS)
    return parser.parse_args(argv)


def _release_baseline_artifacts(
    *,
    bundle_path: Path,
    verification_path: Path,
    output_dir: Path,
    checksum_cache: dict[Path, str],
) -> tuple[list[BundleArtifact], dict[str, int]]:
    """Copy a validated release-baseline bundle and its matching verification."""
    baseline = load_release_baseline_bundle(bundle_path)
    verification_payload = json.loads(verification_path.read_text(encoding="utf-8"))
    verification = release_baseline_verification_from_dict(verification_payload)
    bundle_sha256 = _sha256(bundle_path)
    if verification.bundle_sha256 != bundle_sha256:
        raise ValueError("release baseline verification bundle checksum mismatch")
    if verification.release != baseline.release:
        raise ValueError("release baseline verification release does not match bundle")
    if {
        key: verification.summary[key]
        for key in ("total", "official", "derived", "blocked")
    } != baseline.summary:
        raise ValueError("release baseline verification summary does not match bundle")

    release_dir = output_dir / "release_baseline"
    release_dir.mkdir(parents=True, exist_ok=True)
    copied_bundle = release_dir / "release_baseline_bundle.json"
    copied_verification = release_dir / "release_baseline_verification.json"
    shutil.copyfile(bundle_path, copied_bundle)
    shutil.copyfile(verification_path, copied_verification)
    authority_class = (
        "provisional"
        if baseline.summary["derived"] or baseline.summary["blocked"]
        else "diagnostic-only"
    )
    return (
        [
            _file_artifact(
                id="release_baseline_bundle",
                path=copied_bundle,
                bundle_dir=output_dir,
                authority_class=authority_class,
                status="present",
                description="Complete release baseline evidence bundle.",
                required=True,
                checksum_cache=checksum_cache,
            ),
            _file_artifact(
                id="release_baseline_verification",
                path=copied_verification,
                bundle_dir=output_dir,
                authority_class=authority_class,
                status="present",
                description="Independent release baseline rerun verification.",
                required=True,
                checksum_cache=checksum_cache,
            ),
        ],
        baseline.summary,
    )


def _command_from_args(value: list[str] | None, default: list[str]) -> list[str]:
    return list(value) if value else list(default)


def _expand_command(
    command: list[str],
    *,
    bundle_dir: Path,
    release_validation_dir: Path,
    environment_dir: Path,
) -> list[str]:
    replacements = {
        "{bundle_dir}": str(bundle_dir),
        "{release_validation_dir}": str(release_validation_dir),
        "{environment_dir}": str(environment_dir),
    }
    return [replacements.get(part, part) for part in command]


def _run_command(
    *,
    name: str,
    command: list[str],
    transcript_dir: Path,
    temp_dir: Path,
    log_tail_chars: int,
    failure_classification: str,
    failure_next_action: str,
    stdout_artifact: Path | None = None,
) -> CommandTranscript:
    started = time.monotonic()
    transcript_path = transcript_dir / f"{name}.json"
    stdout_path = _temporary_stream_path(temp_dir, name, "stdout")
    stderr_path = _temporary_stream_path(temp_dir, name, "stderr")
    try:
        completed = _run_command_to_files(command, stdout_path, stderr_path)
        duration_s = time.monotonic() - started
        status = "passed" if completed.returncode == 0 else "failed"
        classification = (
            "diagnostic-only" if status == "passed" else failure_classification
        )
        next_action = "Review recorded artifacts before publishing."
        if status != "passed":
            next_action = failure_next_action
        stdout_tail = _tail_file(stdout_path, log_tail_chars)
        stderr_tail = _tail_file(stderr_path, log_tail_chars)
        if stdout_artifact is not None and completed.returncode == 0:
            stdout_artifact.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stdout_path, stdout_artifact)
        returncode: int | None = completed.returncode
    except FileNotFoundError as exc:
        duration_s = time.monotonic() - started
        status = "unavailable"
        classification = "unavailable"
        next_action = f"Install or expose required command before collecting this evidence: {exc.filename}"
        stdout_tail = ""
        stderr_tail = _tail(str(exc), log_tail_chars)
        returncode = None
    finally:
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)

    transcript_payload = {
        "schema_version": f"{SCHEMA_VERSION}.command_transcript",
        "name": name,
        "command": command,
        "status": status,
        "classification": classification,
        "next_action": next_action,
        "duration_s": round(duration_s, 3),
        "returncode": returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    transcript_path.write_text(
        json.dumps(transcript_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CommandTranscript(
        name=name,
        command=command,
        status=status,
        classification=classification,
        next_action=next_action,
        duration_s=round(duration_s, 3),
        returncode=returncode,
        transcript_path=_relative_path(transcript_path, transcript_dir.parent),
    )


def _release_validation_artifacts(
    release_validation_dir: Path,
    command_status: str,
    *,
    checksum_cache: dict[Path, str],
) -> list[BundleArtifact]:
    status = "present" if command_status == "passed" else command_status
    return [
        _file_artifact(
            id="release_candidate_validation_json",
            path=release_validation_dir / "release_candidate_validation.json",
            bundle_dir=release_validation_dir.parent,
            authority_class="diagnostic-only",
            status=status,
            description="Machine-readable release-candidate validation summary.",
            required=True,
            checksum_cache=checksum_cache,
        ),
        _file_artifact(
            id="release_candidate_validation_markdown",
            path=release_validation_dir / "release_candidate_validation.md",
            bundle_dir=release_validation_dir.parent,
            authority_class="diagnostic-only",
            status=status,
            description="Human-readable release-candidate validation summary.",
            required=True,
            checksum_cache=checksum_cache,
        ),
    ]


def _source_reference_artifacts(
    bundle_dir: Path,
    *,
    checksum_cache: dict[Path, str],
) -> list[BundleArtifact]:
    refs = [
        (
            "claims_doc",
            Path("docs/CLAIMS.md"),
            "diagnostic-only",
            "Claim boundary source document.",
            True,
        ),
        (
            "v1_25_release_notes",
            Path("docs/v1_25_release_notes.md"),
            "provisional",
            "Engineering-prerelease release-note baseline for artifact authority wording.",
            False,
        ),
        (
            "release_candidate_validation_doc",
            Path("docs/release_candidate_validation.md"),
            "diagnostic-only",
            "Release-candidate validation command guide.",
            False,
        ),
    ]
    artifacts: list[BundleArtifact] = []
    for artifact_id, path, authority_class, description, required in refs:
        artifacts.append(
            _file_artifact(
                id=artifact_id,
                path=path,
                bundle_dir=bundle_dir,
                authority_class=authority_class,
                status="present" if path.exists() else "unavailable",
                description=description,
                required=required,
                checksum_cache=checksum_cache,
            )
        )
    return artifacts


def _file_artifact(
    *,
    id: str,
    path: Path,
    bundle_dir: Path,
    authority_class: str,
    status: str,
    description: str,
    required: bool = False,
    checksum_cache: dict[Path, str],
) -> BundleArtifact:
    if not path.exists():
        return BundleArtifact(
            id=id,
            path=_safe_relative_path(path, bundle_dir),
            authority_class=authority_class,
            status="unavailable" if status == "present" else status,
            description=description,
            required=required,
        )
    return BundleArtifact(
        id=id,
        path=_safe_relative_path(path, bundle_dir),
        authority_class=authority_class,
        status=status,
        description=description,
        sha256=_sha256_cached(path, checksum_cache),
        required=required,
    )


def _authority_map() -> list[dict[str, object]]:
    return [
        {
            "id": "trace_jsonl",
            "authority_class": "canonical",
            "status": "deferred",
            "description": "Trace JSONL remains the canonical run artifact; this bundle does not generate traces by default.",
        },
        {
            "id": "release_validation_and_transcripts",
            "authority_class": "diagnostic-only",
            "status": "present",
            "description": "Release validation output, environment evidence, and transcripts are review evidence only.",
        },
        {
            "id": "bounded_dataset_slice",
            "authority_class": "provisional",
            "status": "deferred",
            "description": "Bounded slice evidence is useful for prerelease review but not paper parity.",
        },
        {
            "id": "full_235_problem_validation",
            "authority_class": "deferred",
            "status": "deferred",
            "description": "Paper-scale validation is outside this prerelease artifact bundle.",
        },
        {
            "id": "cdna4_validation",
            "authority_class": "unavailable",
            "status": "unavailable",
            "description": "CDNA4 validation is unavailable because suitable hardware is not currently accessible.",
        },
    ]


def _default_known_gaps() -> list[dict[str, str]]:
    return [
        {
            "id": "mi300x_cdna3_full_suite",
            "status": "deferred",
            "description": (
                "CDNA3/gfx942 validation infrastructure evidence was recorded "
                "on MI308X, not MI300X; full-suite MI300X validation remains "
                "deferred."
            ),
        },
        {
            "id": "cdna4_validation",
            "status": "unavailable",
            "description": "CDNA4 validation is unavailable because suitable hardware is not currently accessible.",
        },
        {
            "id": "full_235_problem_validation",
            "status": "deferred",
            "description": "Full paper-scale validation is not generated by the default prerelease bundle.",
        },
    ]


def _claim_boundary() -> dict[str, bool]:
    return {
        "engineering_prerelease_only": True,
        "research_preview_only": True,
        "full_235_problem_validation": False,
        "upstream_solar_parity": False,
        "leaderboard_ready": False,
        "hard_sandbox": False,
        "native_host_validation_from_docker": False,
        "mi300x_cdna3_full_suite_validated": False,
        "cdna4_validated": False,
        "release_baseline_full_suite_official": False,
    }


def _summary(
    transcripts: list[CommandTranscript],
    artifacts: list[BundleArtifact],
) -> dict[str, object]:
    return {
        "commands": len(transcripts),
        "command_statuses": {
            "passed": sum(transcript.status == "passed" for transcript in transcripts),
            "failed": sum(transcript.status == "failed" for transcript in transcripts),
            "unavailable": sum(
                transcript.status == "unavailable" for transcript in transcripts
            ),
        },
        "artifacts": len(artifacts),
        "artifact_statuses": {
            "present": sum(artifact.status == "present" for artifact in artifacts),
            "deferred": sum(artifact.status == "deferred" for artifact in artifacts),
            "unavailable": sum(
                artifact.status == "unavailable" for artifact in artifacts
            ),
            "failed": sum(artifact.status == "failed" for artifact in artifacts),
        },
        "authority_classes": sorted(
            {artifact.authority_class for artifact in artifacts}
            | {entry["authority_class"] for entry in _authority_map()}
        ),
    }


def _overall_status(
    transcripts: list[CommandTranscript],
    artifacts: list[BundleArtifact],
) -> str:
    blocking_failed = any(
        transcript.classification == "blocking" and transcript.status != "passed"
        for transcript in transcripts
    )
    required_missing = any(
        artifact.required and artifact.status not in {"present"}
        for artifact in artifacts
    )
    if blocking_failed or required_missing:
        return "blocking"
    if any(
        artifact.status in {"deferred", "unavailable", "failed"}
        for artifact in artifacts
    ) or any(transcript.status != "passed" for transcript in transcripts):
        return "review_needed"
    return "passed"


def _git_context(source_ref: str | None) -> dict[str, object]:
    commit = _git_output(["git", "rev-parse", "HEAD"])
    tag = source_ref or _git_output(["git", "describe", "--tags", "--exact-match"])
    status = _git_output(["git", "status", "--short"])
    return {
        "commit": commit or None,
        "source_ref": tag or source_ref,
        "clean_tree": status == "",
    }


def _git_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _temporary_stream_path(temp_dir: Path, name: str, stream_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    with tempfile.NamedTemporaryFile(
        prefix=f"{safe_name}_{stream_name}_",
        suffix=".log",
        dir=temp_dir,
        delete=False,
    ) as handle:
        return Path(handle.name)


def _run_command_to_files(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[str]:
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open(
            "w",
            encoding="utf-8",
        ) as stderr_handle,
    ):
        completed = subprocess.run(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    if completed.stdout:
        stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        stderr_path.write_text(completed.stderr, encoding="utf-8")
    return completed


def _validate_authority_classes(
    artifacts: list[BundleArtifact],
    authority_map: list[dict[str, object]],
) -> None:
    classes = {artifact.authority_class for artifact in artifacts}
    classes.update(str(entry["authority_class"]) for entry in authority_map)
    unknown = classes - AUTHORITY_CLASSES
    if unknown:
        raise ValueError(f"unknown authority class: {sorted(unknown)}")
    missing = AUTHORITY_CLASSES - classes
    if missing:
        raise ValueError(f"authority map missing required classes: {sorted(missing)}")


def _write_checksums(bundle_dir: Path, checksum_cache: dict[Path, str]) -> None:
    checksum_path = bundle_dir / "SHA256SUMS"
    files = sorted(
        path
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name != checksum_path.name
    )
    _sha256_cached_many(files, checksum_cache)
    lines = [
        f"{_sha256_cached(path, checksum_cache)}  {_relative_path(path, bundle_dir)}"
        for path in files
    ]
    checksum_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


def _render_markdown(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, dict)
    lines = [
        "# Prerelease Artifact Bundle",
        "",
        f"**Schema:** `{payload['schema_version']}`",
        f"**Version:** `{payload['bundle_version']}`",
        f"**Overall status:** `{payload['overall_status']}`",
        "",
        "## Claim Boundary",
        "",
        "This bundle is engineering prerelease and research preview evidence only. "
        "It is not full 235-problem paper validation, upstream SOLAR parity, "
        "leaderboard readiness, hard-sandbox evidence, native-host validation "
        "from Docker evidence, full validation of the MI300X GPU model under CDNA3, "
        "or CDNA4 validation.",
        "",
        "## Authority Classes",
        "",
        "| Surface | Authority class | Status |",
        "| --- | --- | --- |",
    ]
    for entry in payload["authority_map"]:
        assert isinstance(entry, dict)
        lines.append(
            f"| `{entry['id']}` | `{entry['authority_class']}` | `{entry['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "| Command | Status | Classification | Transcript |",
            "| --- | --- | --- | --- |",
        ]
    )
    for command in payload["commands"]:
        assert isinstance(command, dict)
        lines.append(
            f"| `{command['name']}` | `{command['status']}` | "
            f"`{command['classification']}` | `{command['transcript_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| Artifact | Authority class | Status | SHA-256 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for artifact in payload["artifacts"]:
        assert isinstance(artifact, dict)
        sha = artifact.get("sha256") or ""
        lines.append(
            f"| `{artifact['id']}` | `{artifact['authority_class']}` | "
            f"`{artifact['status']}` | `{sha}` |"
        )
    lines.extend(
        [
            "",
            "## Checksums",
            "",
            "Review `SHA256SUMS` for the checksums of generated and referenced files.",
            "",
        ]
    )
    return "\n".join(lines)


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


def _tail(value: str, limit: int = DEFAULT_LOG_TAIL_CHARS) -> str:
    if limit <= 0:
        return ""
    redacted = TOKEN_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>", value
    )
    return redacted[-limit:]


def _append_tail(tail: str, text: str, limit: int) -> str:
    if not text:
        return tail
    return (tail + text)[-limit:]


def _redacted_tail_file(path: Path, limit: int) -> str:
    tail = ""
    pending = ""
    in_secret = False
    chunk_size = max(limit, 8192)

    with path.open("rb") as handle:
        for raw_chunk in iter(lambda: handle.read(chunk_size), b""):
            text = raw_chunk.decode(errors="replace")
            if in_secret:
                delimiter_index = next(
                    (
                        index
                        for index, char in enumerate(text)
                        if char in _TOKEN_VALUE_DELIMITERS
                    ),
                    None,
                )
                if delimiter_index is None:
                    continue
                tail = _append_tail(tail, text[delimiter_index], limit)
                text = text[delimiter_index + 1 :]
                in_secret = False

            pending += text
            while pending:
                match = TOKEN_PATTERN.search(pending)
                if match is None:
                    if len(pending) > _TOKEN_PREFIX_OVERLAP_CHARS:
                        emit = pending[:-_TOKEN_PREFIX_OVERLAP_CHARS]
                        tail = _append_tail(tail, emit, limit)
                        pending = pending[-_TOKEN_PREFIX_OVERLAP_CHARS:]
                    break

                tail = _append_tail(tail, pending[: match.start()], limit)
                tail = _append_tail(
                    tail,
                    f"{match.group(1)}{match.group(2)}<redacted>",
                    limit,
                )
                if match.end() == len(pending):
                    pending = ""
                    in_secret = True
                    break
                pending = pending[match.end() :]

    if not in_secret:
        tail = _append_tail(tail, _tail(pending, max(limit, 8192)), limit)
    return tail


def _tail_file(path: Path, limit: int = DEFAULT_LOG_TAIL_CHARS) -> str:
    if limit <= 0:
        return ""
    try:
        return _redacted_tail_file(path, limit)
    except OSError:
        return ""


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
