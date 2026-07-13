#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build a strict publication manifest from a self-contained authority source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.scoring.release_baseline import (
    CandidateIdentity,
    EvidencePublicationManifest,
    PublishedArtifact,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--artifact-base-uri", required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()

    bundle = _load(root / "release-baseline-bundle.json")
    verification = _load(root / "release-baseline-verification.json")
    evidence = _load(root / "score/official-score-evidence.json")
    artifacts: dict[str, PublishedArtifact] = {}

    def add(role: str, relative_path: str) -> None:
        if relative_path in artifacts:
            return
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"missing publication artifact: {relative_path}")
        artifacts[relative_path] = PublishedArtifact(
            role=role,
            relative_path=relative_path,
            sha256=sha256_file(path),
        )

    fixed = {
        "candidate_solution": "candidate/solution.json",
        "candidate_trace": "candidate/trace.jsonl",
        "candidate_timing": "candidate/timing-evidence.jsonl",
        "scoring_baseline": "scoring-baseline.json",
        "release_baseline_bundle": "release-baseline-bundle.json",
        "release_baseline_verification": "release-baseline-verification.json",
        "suite_manifest": "suite/authority-suite-manifest.json",
        "official_score_evidence": "score/official-score-evidence.json",
        "amd_native_score": "score/amd-native-score.json",
        "authority_input": "authority.json",
        "baseline_rerun_trace": str(verification["rerun_trace_ref"]),
        "hardware_calibration_primary": "out/gfx1200-full-suite-closure/calibration-primary-v3.json",
        "hardware_calibration_verification": "out/gfx1200-full-suite-closure/calibration-verification-v3.json",
        "hardware_profile_requirements": "out/gfx1200-full-suite-closure/hardware-profile-requirements.json",
    }
    for role, relative_path in fixed.items():
        add(role, relative_path)

    workloads = bundle.get("workloads")
    if not isinstance(workloads, list):
        raise ValueError("release bundle must contain workloads")
    model_refs: set[str] = set()
    for row in workloads:
        if not isinstance(row, dict) or row.get("classification") != "official":
            raise ValueError("authority publication bundle must be entirely official")
        definition = str(row["definition"])
        workload_uuid = str(row["workload_uuid"])
        bound_ref = str(row["bound_ref"])
        model_ref = str(row["hardware_model_ref"])
        add(f"amd_sol_bound:{definition}:{workload_uuid}", bound_ref)
        model_refs.add(model_ref)
        bound = _load(root / bound_ref)
        fusion_ref = bound.get("fusion_validation_ref")
        if not isinstance(fusion_ref, str):
            raise ValueError(f"{bound_ref} has no fusion validation reference")
        fusion_path = (Path(bound_ref).parent / fusion_ref).as_posix()
        add(f"fusion_validation:{fusion_path}", fusion_path)
    for model_ref in sorted(model_refs):
        add(f"hardware_model:{model_ref}", model_ref)

    candidate = evidence.get("candidate_evidence")
    if not isinstance(candidate, dict):
        raise ValueError("official score evidence has no candidate identity")
    manifest = EvidencePublicationManifest(
        release=str(bundle["release"]),
        scope=str(bundle["scope"]),
        source_repository=args.source_repository,
        source_revision=args.source_revision,
        container_image_digest=args.container_image_digest,
        artifact_base_uri=args.artifact_base_uri,
        candidate=CandidateIdentity(
            solution_ref="candidate/solution.json",
            solution_sha256=sha256_file(root / "candidate/solution.json"),
            trace_relative_path="candidate/trace.jsonl",
            trace_sha256=sha256_file(root / "candidate/trace.jsonl"),
            timing_relative_path="candidate/timing-evidence.jsonl",
            timing_sha256=sha256_file(root / "candidate/timing-evidence.jsonl"),
        ),
        artifacts=tuple(artifacts[path] for path in sorted(artifacts)),
    ).with_checksum()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"artifacts": len(artifacts), "output": str(args.output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
