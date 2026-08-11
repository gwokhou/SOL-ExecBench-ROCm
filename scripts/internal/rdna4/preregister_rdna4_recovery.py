#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Preregister a corrected design while reusing a proven-stable VRAM policy."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

from sol_execbench.core.bench.performance_model.corpus_preflight import (
    DiagnosticCorpusDesign,
)
from sol_execbench.core.bench.performance_model.lifecycle import (
    design_id,
    load_and_verify_source_review,
)
from sol_execbench.core.bench.performance_model.source_transition import (
    DiagnosticSourceTransitionAttestation,
    SourceTransitionDisposition,
    SourceTransitionStage,
)
from sol_execbench.core.bench.performance_model.vram_policy import (
    DiagnosticVRAMWorkingSetPolicy,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import sha256_file

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_COLLECTOR_PATH = (
    _REPOSITORY_ROOT
    / "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
)
_TRANSITION_TOOL = (
    _REPOSITORY_ROOT
    / "scripts/internal/rdna4/manage_rdna4_source_transition.py"
)


def _load_collector() -> ModuleType:
    name = "_sol_execbench_rdna4_recovery_collector"
    spec = spec_from_file_location(name, _COLLECTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load collector module: {_COLLECTOR_PATH}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


collector = _load_collector()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--universe-start", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--vram-policy", type=Path, required=True)
    parser.add_argument("--prior-attestation", type=Path, required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    return parser.parse_args()


def _verify_current_source(revision: str) -> None:
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()
    if observed != revision:
        raise ValueError("recovery source revision is not current HEAD")
    dirty = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src",
            "scripts",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    if dirty:
        raise ValueError("recovery source paths contain uncommitted changes")


def _verify_prior_attestation(
    path: Path,
) -> DiagnosticSourceTransitionAttestation:
    subprocess.run(
        [
            sys.executable,
            str(_TRANSITION_TOOL),
            "verify",
            "--attestation",
            str(path.resolve()),
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return load_json_file(DiagnosticSourceTransitionAttestation, path)


def _require_reusable_policy_chain(
    *,
    policy_path: Path,
    source_revision: str,
    prior: DiagnosticSourceTransitionAttestation,
    review_path: Path,
) -> DiagnosticVRAMWorkingSetPolicy:
    review = load_and_verify_source_review(
        review_path,
        repository_root=_REPOSITORY_ROOT,
    )
    if (
        prior.target_source_revision != review.base_source_revision
        or review.target_source_revision != source_revision
    ):
        raise ValueError(
            "policy reuse source-transition chain is discontinuous"
        )
    prior_decisions = {
        item.stage: item.disposition for item in prior.stage_decisions
    }
    protected_stages = (
        SourceTransitionStage.VRAM_POLICY,
        SourceTransitionStage.CALIBRATION,
    )
    if any(
        prior_decisions[stage] is not SourceTransitionDisposition.UNCHANGED
        or review.affects(stage)
        for stage in protected_stages
    ):
        raise ValueError(
            "policy/calibration semantics changed across reuse chain"
        )
    policy = load_json_file(DiagnosticVRAMWorkingSetPolicy, policy_path)
    if policy.source_revision != prior.base_source_revision:
        raise ValueError("reused policy is not the attested base policy")
    expected = next(
        (
            artifact.sha256
            for artifact in prior.reusable_artifacts
            if artifact.relative_path == "vram-policy.json"
        ),
        None,
    )
    if expected is None or sha256_file(policy_path) != expected:
        raise ValueError(
            "reused VRAM policy differs from transition attestation"
        )
    return policy


def _preregister(arguments: argparse.Namespace) -> tuple[str, str]:
    _verify_current_source(arguments.source_revision)
    prior = _verify_prior_attestation(arguments.prior_attestation)
    policy = _require_reusable_policy_chain(
        policy_path=arguments.vram_policy,
        source_revision=arguments.source_revision,
        prior=prior,
        review_path=arguments.source_review,
    )
    root = arguments.root.resolve()
    design_path = root / "design.json"
    policy_path = root / "vram-policy.json"
    if design_path.exists() or policy_path.exists():
        raise ValueError(
            "recovery preregistration refuses existing design inputs"
        )
    collector._validate_design_working_sets(arguments.universe_start, policy)
    payload = collector._design_payload(
        arguments.universe_start,
        sha256_file(arguments.vram_policy),
    )
    design = DiagnosticCorpusDesign.model_validate(payload)
    collector._validate_design_contracts(
        collector._all_cases(arguments.universe_start)
    )
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(arguments.vram_policy, policy_path)
    atomic_write_json_value(design_path, design.model_dump(mode="json"))
    design_digest = sha256_file(design_path)
    policy_digest = sha256_file(policy_path)
    stage_id = design_id(
        universe_start=arguments.universe_start,
        design_payload_sha256=design_digest,
        source_revision=arguments.source_revision,
        vram_policy_sha256=policy_digest,
    )
    collector._write_design_manifest(
        root=root,
        universe_start=arguments.universe_start,
        design_payload_sha256=design_digest,
        did=stage_id,
        source_revision=arguments.source_revision,
        vram_policy_sha256=policy_digest,
    )
    return stage_id, design_digest


def main() -> int:
    """Run the reviewed recovery preregistration."""
    arguments = _parse_args()
    stage_id, design_digest = _preregister(arguments)
    print(
        json.dumps(
            {"design_id": stage_id, "design_sha256": design_digest},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
