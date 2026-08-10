#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Govern acceptance exposure and case-granular held-out reuse.

This control-plane tool is intentionally separate from the GPU collector so
adding reuse policy cannot itself change the collector identity cited by old
raw evidence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

from sol_execbench.core.bench.performance_model.case_reuse import (
    CASE_REUSE_MANIFEST_NAME,
    EXPOSURE_RECEIPT_NAME,
    REPLACEMENT_FRAGMENT_NAME,
    SOURCE_CORPUS_NAME,
    DiagnosticAcceptanceExposureReceipt,
    DiagnosticCaseReuseManifest,
    DiagnosticHeldOutCorpusFragment,
    SourceChangeImpact,
    build_case_reuse_decisions,
    compose_case_reuse_corpus,
    load_and_verify_case_reuse_bundle,
    persist_acceptance_exposure,
)
from sol_execbench.core.bench.performance_model.lifecycle import (
    DiagnosticLifecycleStage,
    DiagnosticStageAttempt,
    store_root,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.validation_corpus import (
    DiagnosticValidationCorpus,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_file

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _load_collector() -> ModuleType:
    """Load the adjacent production collector without changing its module."""
    path = Path(__file__).with_name("build_rdna4_diagnostic_corpora.py")
    name = "_sol_execbench_rdna4_diagnostic_collector"
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load collector module: {path}")
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
    parser.add_argument(
        "stage",
        choices=("freeze-fragment", "compose-held-out", "record-exposure"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--family", choices=tuple(item.value for item in collector.FAMILIES)
    )
    parser.add_argument("--source-corpus", type=Path)
    parser.add_argument("--replacement-fragment", type=Path)
    parser.add_argument("--exposure-receipt", type=Path)
    parser.add_argument("--impact-review", type=Path)
    parser.add_argument("--base-source-revision")
    parser.add_argument("--target-source-revision")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--held-out-corpus", type=Path)
    parser.add_argument("--released-case-id")
    parser.add_argument("--reason-code", action="append", default=[])
    parser.add_argument(
        "--replace-family",
        action="append",
        default=[],
        choices=tuple(item.value for item in collector.FAMILIES),
    )
    parser.add_argument("--source-revision")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _freeze_fragment(root: Path, family: WorkloadKind, output: Path) -> None:
    """Freeze one fully evidenced held-out family without freezing 220 cases."""
    if output.exists():
        raise ValueError(f"refusing to overwrite held-out fragment: {output}")
    design = collector._require_frozen_design(root)
    cases = [
        collector._validation_case(root, case)
        for case in collector._cases("held_out", design.universe_start)
        if case.family is family
    ]
    fragment = DiagnosticHeldOutCorpusFragment(
        design_sha256=sha256_file(root / "design.json"),
        cases=cases,
    )
    atomic_write_json_value(output, fragment.model_dump(mode="json"))


def _load_impact_review(path: Path) -> tuple[SourceChangeImpact, ...]:
    raw = load_json_value(path)
    if not isinstance(raw, list):
        raise ValueError("impact review must be a JSON list")
    review = tuple(SourceChangeImpact.model_validate(item) for item in raw)
    paths = tuple(item.path for item in review)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise ValueError("impact review paths must be sorted and unique")
    return review


def _git_change_identity(line: str) -> tuple[str, str | None, str]:
    fields = line.split("\t")
    status = fields[0][0]
    change = {
        "A": "added",
        "D": "deleted",
        "M": "modified",
        "R": "renamed",
    }.get(status)
    if change is None:
        raise ValueError(f"unsupported git change status: {fields[0]}")
    if status == "R" and len(fields) == 3:
        return change, fields[1], fields[2]
    if len(fields) != 2:
        raise ValueError(f"malformed git change record: {line}")
    return change, None, fields[1]


def _verify_impact_review(
    base_revision: str,
    target_revision: str,
    review: tuple[SourceChangeImpact, ...],
) -> None:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "--find-renames",
            base_revision,
            target_revision,
            "--",
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot verify reviewed source diff: {result.stderr}")
    actual = tuple(
        sorted(
            (_git_change_identity(line) for line in result.stdout.splitlines()),
            key=lambda item: item[2],
        )
    )
    declared = tuple(
        (item.change, item.previous_path, item.path) for item in review
    )
    if declared != actual:
        raise ValueError(
            "impact review does not exactly cover the version diff"
        )


def _require_compose_inputs(arguments: argparse.Namespace) -> None:
    required = {
        "--source-corpus": arguments.source_corpus,
        "--replacement-fragment": arguments.replacement_fragment,
        "--exposure-receipt": arguments.exposure_receipt,
        "--impact-review": arguments.impact_review,
        "--base-source-revision": arguments.base_source_revision,
        "--target-source-revision": arguments.target_source_revision,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"compose-held-out requires {', '.join(missing)}")
    if not arguments.replace_family:
        raise ValueError("compose-held-out requires --replace-family")


def _write_reuse_inputs(
    output: Path,
    source: DiagnosticValidationCorpus,
    fragment: DiagnosticHeldOutCorpusFragment,
    exposure: DiagnosticAcceptanceExposureReceipt,
) -> tuple[Path, Path, Path]:
    output.mkdir(parents=True, exist_ok=False)
    paths = (
        output / SOURCE_CORPUS_NAME,
        output / REPLACEMENT_FRAGMENT_NAME,
        output / EXPOSURE_RECEIPT_NAME,
    )
    for path, model in zip(paths, (source, fragment, exposure), strict=True):
        atomic_write_json_value(path, model.model_dump(mode="json"))
    return paths


def _compose_held_out(arguments: argparse.Namespace) -> None:
    _require_compose_inputs(arguments)
    source = load_json_file(DiagnosticValidationCorpus, arguments.source_corpus)
    fragment = load_json_file(
        DiagnosticHeldOutCorpusFragment, arguments.replacement_fragment
    )
    exposure = load_json_file(
        DiagnosticAcceptanceExposureReceipt, arguments.exposure_receipt
    )
    tainted = tuple(WorkloadKind(item) for item in arguments.replace_family)
    final = compose_case_reuse_corpus(source, fragment, tainted)
    source_path, fragment_path, exposure_path = _write_reuse_inputs(
        arguments.output.resolve(), source, fragment, exposure
    )
    final_path = arguments.output.resolve() / "held_out.json"
    atomic_write_json_value(final_path, final.model_dump(mode="json"))
    changes = _load_impact_review(arguments.impact_review)
    _verify_impact_review(
        arguments.base_source_revision,
        arguments.target_source_revision,
        changes,
    )
    manifest = DiagnosticCaseReuseManifest(
        source_corpus_sha256=sha256_file(source_path),
        replacement_fragment_sha256=sha256_file(fragment_path),
        replacement_design_sha256=fragment.design_sha256,
        exposure_receipt_sha256=sha256_file(exposure_path),
        final_corpus_sha256=sha256_file(final_path),
        base_source_revision=arguments.base_source_revision,
        target_source_revision=arguments.target_source_revision,
        source_changes=changes,
        tainted_families=tainted,
        decisions=build_case_reuse_decisions(final, source, tainted),
        created_at=datetime.now(UTC).isoformat(),
    )
    atomic_write_json_value(
        arguments.output.resolve() / CASE_REUSE_MANIFEST_NAME,
        manifest.model_dump(mode="json"),
    )
    load_and_verify_case_reuse_bundle(final_path)


def _record_exposure(arguments: argparse.Namespace) -> None:
    required = {
        "--attempt": arguments.attempt,
        "--held-out-corpus": arguments.held_out_corpus,
        "--released-case-id": arguments.released_case_id,
        "--source-revision": arguments.source_revision,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing or not arguments.reason_code:
        detail = ", ".join(missing or ["--reason-code"])
        raise ValueError(f"record-exposure requires {detail}")
    attempt = load_json_file(DiagnosticStageAttempt, arguments.attempt)
    if attempt.stage is not DiagnosticLifecycleStage.ACCEPTANCE:
        raise ValueError("exposure source is not an acceptance attempt")
    corpus = load_json_file(
        DiagnosticValidationCorpus, arguments.held_out_corpus
    )
    selected = [
        (index, case)
        for index, case in enumerate(corpus.cases)
        if case.case_id == arguments.released_case_id
    ]
    if len(selected) != 1:
        raise ValueError("released case is not unique in held-out corpus")
    index, case = selected[0]
    if case.case_id not in attempt.detail or any(
        reason not in attempt.detail for reason in arguments.reason_code
    ):
        raise ValueError("attempt detail does not prove the declared exposure")
    receipt = DiagnosticAcceptanceExposureReceipt(
        purpose=corpus.purpose,
        run_id=attempt.run_id,
        held_out_corpus_sha256=sha256_file(arguments.held_out_corpus),
        source_revision=arguments.source_revision,
        evaluated_case_ids_before_failure=tuple(
            item.case_id for item in corpus.cases[:index]
        ),
        released_case_id=case.case_id,
        released_workload_kind=case.workload_kind,
        released_reason_codes=tuple(arguments.reason_code),
        created_at=attempt.finished_at,
    )
    _write_and_register_exposure(receipt, arguments.output)


def _write_and_register_exposure(
    receipt: DiagnosticAcceptanceExposureReceipt,
    output: Path,
) -> None:
    if output.exists():
        existing = load_json_file(DiagnosticAcceptanceExposureReceipt, output)
        if existing != receipt:
            raise ValueError(f"immutable exposure differs: {output}")
    else:
        atomic_write_json_value(output, receipt.model_dump(mode="json"))
    digest = persist_acceptance_exposure(receipt, output, store_root())
    print(f"recorded acceptance exposure {digest} at {output}")


def main() -> int:
    """Run one exposure, fragment-freeze, or corpus-composition stage."""
    arguments = _parse_args()
    if arguments.stage == "freeze-fragment":
        if arguments.family is None:
            raise ValueError("freeze-fragment requires --family")
        _freeze_fragment(
            arguments.root.resolve(),
            WorkloadKind(arguments.family),
            arguments.output.resolve(),
        )
    elif arguments.stage == "compose-held-out":
        _compose_held_out(arguments)
    else:
        _record_exposure(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
