#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Build a versioned prerelease artifact bundle for review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

SCHEMA_VERSION = "sol_execbench.prerelease_artifact_bundle.v1"
DEFAULT_OUTPUT_DIR = Path("out/prerelease_artifact_bundle")
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
    "scripts/release_candidate_validation.py",
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
    for path in (bundle_dir, release_validation_dir, transcript_dir, environment_dir):
        path.mkdir(parents=True, exist_ok=True)

    transcripts: list[CommandTranscript] = []
    artifacts: list[BundleArtifact] = []
    known_gaps = _default_known_gaps()

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
            log_tail_chars=args.log_tail_chars,
            failure_classification="blocking",
            failure_next_action="Fix release-candidate validation before publishing.",
        )
        transcripts.append(result)
        artifacts.extend(_release_validation_artifacts(release_validation_dir, result.status))

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
            )
        )

    artifacts.extend(_source_reference_artifacts(bundle_dir))
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

    manifest_path = bundle_dir / "prerelease_artifact_bundle.json"
    markdown_path = bundle_dir / "prerelease_artifact_bundle.md"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    _write_checksums(bundle_dir)
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
    parser.add_argument("--log-tail-chars", type=int, default=DEFAULT_LOG_TAIL_CHARS)
    return parser.parse_args(argv)


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
    log_tail_chars: int,
    failure_classification: str,
    failure_next_action: str,
    stdout_artifact: Path | None = None,
) -> CommandTranscript:
    started = time.monotonic()
    transcript_path = transcript_dir / f"{name}.json"
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        duration_s = time.monotonic() - started
        status = "passed" if completed.returncode == 0 else "failed"
        classification = "diagnostic-only" if status == "passed" else failure_classification
        next_action = "Review recorded artifacts before publishing."
        if status != "passed":
            next_action = failure_next_action
        stdout_tail = _tail(completed.stdout, log_tail_chars)
        stderr_tail = _tail(completed.stderr, log_tail_chars)
        if stdout_artifact is not None and completed.returncode == 0:
            stdout_artifact.write_text(completed.stdout, encoding="utf-8")
        returncode: int | None = completed.returncode
    except FileNotFoundError as exc:
        duration_s = time.monotonic() - started
        status = "unavailable"
        classification = "unavailable"
        next_action = f"Install or expose required command before collecting this evidence: {exc.filename}"
        stdout_tail = ""
        stderr_tail = _tail(str(exc), log_tail_chars)
        returncode = None

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
        ),
        _file_artifact(
            id="release_candidate_validation_markdown",
            path=release_validation_dir / "release_candidate_validation.md",
            bundle_dir=release_validation_dir.parent,
            authority_class="diagnostic-only",
            status=status,
            description="Human-readable release-candidate validation summary.",
            required=True,
        ),
    ]


def _source_reference_artifacts(bundle_dir: Path) -> list[BundleArtifact]:
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
        sha256=_sha256(path),
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
                "MI300X is the concrete CDNA3 `gfx942` hardware target, "
                "but full-suite MI300X validation remains deferred."
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
            "unavailable": sum(transcript.status == "unavailable" for transcript in transcripts),
        },
        "artifacts": len(artifacts),
        "artifact_statuses": {
            "present": sum(artifact.status == "present" for artifact in artifacts),
            "deferred": sum(artifact.status == "deferred" for artifact in artifacts),
            "unavailable": sum(artifact.status == "unavailable" for artifact in artifacts),
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
        artifact.required and artifact.status not in {"present"} for artifact in artifacts
    )
    if blocking_failed or required_missing:
        return "blocking"
    if any(
        artifact.status in {"deferred", "unavailable", "failed"} for artifact in artifacts
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


def _write_checksums(bundle_dir: Path) -> None:
    checksum_path = bundle_dir / "SHA256SUMS"
    files = sorted(
        path
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name != checksum_path.name
    )
    lines = [f"{_sha256(path)}  {_relative_path(path, bundle_dir)}" for path in files]
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
        "from Docker evidence, full MI300X validation on the CDNA3 `gfx942` target, "
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail(value: str, limit: int = DEFAULT_LOG_TAIL_CHARS) -> str:
    if limit <= 0:
        return ""
    redacted = TOKEN_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", value)
    return redacted[-limit:]


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
