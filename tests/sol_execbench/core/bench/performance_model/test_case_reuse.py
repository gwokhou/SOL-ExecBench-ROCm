"""Case-granular exposure and held-out evidence reuse contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

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
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.validation_corpus import (
    BlobArtifactReference,
    DiagnosticValidationCase,
    DiagnosticValidationCorpus,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file

_NOW = "2026-08-10T00:00:00+00:00"
_FAMILIES = tuple(
    family for family in WorkloadKind if family is not WorkloadKind.UNSUPPORTED
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _case(
    family: WorkloadKind, index: int, *, fresh: bool
) -> DiagnosticValidationCase:
    family_index = list(_FAMILIES).index(family)
    offset = 10_000 if fresh else 0
    identity = offset + family_index * 100 + index + 1
    return DiagnosticValidationCase(
        case_id=f"held_out-{family.value}-{index:02d}",
        pair_id=_digest(identity),
        workload_kind=family,
        evidence_manifest=BlobArtifactReference(
            sha256=_digest(identity + 20_000),
            size_bytes=100,
            tree_manifest_sha256=_digest(identity + 30_000),
        ),
        solar_manifest=BlobArtifactReference(
            sha256=_digest(identity + 40_000),
            size_bytes=100,
            tree_manifest_sha256=_digest(identity + 50_000),
        ),
    )


def _source() -> DiagnosticValidationCorpus:
    return DiagnosticValidationCorpus(
        role="held_out",
        cases=[
            _case(family, index, fresh=False)
            for family in _FAMILIES
            for index in range(20)
        ],
    )


def _fragment() -> DiagnosticHeldOutCorpusFragment:
    return DiagnosticHeldOutCorpusFragment(
        design_sha256="d" * 64,
        cases=[
            _case(WorkloadKind.ELEMENTWISE, index, fresh=True)
            for index in range(20)
        ],
    )


def _exposure(source_sha256: str) -> DiagnosticAcceptanceExposureReceipt:
    return DiagnosticAcceptanceExposureReceipt(
        run_id="a" * 64,
        held_out_corpus_sha256=source_sha256,
        source_revision="b" * 40,
        evaluated_case_ids_before_failure=("held_out-elementwise-00",),
        released_case_id="held_out-elementwise-01",
        released_workload_kind=WorkloadKind.ELEMENTWISE,
        released_reason_codes=("calibration_out_of_range:working_set_bytes",),
        created_at=_NOW,
    )


def _write_bundle(root: Path) -> Path:
    source = _source()
    fragment = _fragment()
    source_path = root / SOURCE_CORPUS_NAME
    fragment_path = root / REPLACEMENT_FRAGMENT_NAME
    atomic_write_json_value(source_path, source.model_dump(mode="json"))
    atomic_write_json_value(fragment_path, fragment.model_dump(mode="json"))
    exposure = _exposure(sha256_file(source_path))
    exposure_path = root / EXPOSURE_RECEIPT_NAME
    atomic_write_json_value(exposure_path, exposure.model_dump(mode="json"))
    final = compose_case_reuse_corpus(
        source, fragment, (WorkloadKind.ELEMENTWISE,)
    )
    final_path = root / "held_out.json"
    atomic_write_json_value(final_path, final.model_dump(mode="json"))
    manifest = DiagnosticCaseReuseManifest(
        source_corpus_sha256=sha256_file(source_path),
        replacement_fragment_sha256=sha256_file(fragment_path),
        replacement_design_sha256=fragment.design_sha256,
        exposure_receipt_sha256=sha256_file(exposure_path),
        final_corpus_sha256=sha256_file(final_path),
        base_source_revision="b" * 40,
        target_source_revision="c" * 40,
        source_changes=(
            SourceChangeImpact(
                path="HANDSOFF.md",
                change="modified",
                affects_raw_collection=False,
                affects_derived_diagnostics=False,
                rationale="documentation-only change",
            ),
        ),
        tainted_families=(WorkloadKind.ELEMENTWISE,),
        decisions=build_case_reuse_decisions(
            final, source, (WorkloadKind.ELEMENTWISE,)
        ),
        created_at=_NOW,
    )
    atomic_write_json_value(
        root / CASE_REUSE_MANIFEST_NAME, manifest.model_dump(mode="json")
    )
    return final_path


def test_bundle_reuses_200_cases_and_replaces_exposed_family(
    tmp_path: Path,
) -> None:
    final_path = _write_bundle(tmp_path)

    manifest = load_and_verify_case_reuse_bundle(final_path)

    assert manifest is not None
    reused = [
        item for item in manifest.decisions if item.disposition == "reuse"
    ]
    replaced = [
        item for item in manifest.decisions if item.disposition == "replace"
    ]
    assert len(reused) == 200
    assert len(replaced) == 20
    assert {item.workload_kind for item in replaced} == {
        WorkloadKind.ELEMENTWISE
    }


def test_bundle_rejects_replacement_pair_from_exposed_source() -> None:
    source = _source()
    fragment = _fragment().model_copy(
        update={
            "cases": [
                source.cases[0],
                *_fragment().cases[1:],
            ]
        }
    )

    with pytest.raises(ValueError, match="reuses an exposed pair"):
        compose_case_reuse_corpus(source, fragment, (WorkloadKind.ELEMENTWISE,))


def test_bundle_rejects_exposure_prefix_drift(tmp_path: Path) -> None:
    final_path = _write_bundle(tmp_path)
    exposure_path = tmp_path / EXPOSURE_RECEIPT_NAME
    exposure = DiagnosticAcceptanceExposureReceipt.model_validate_json(
        exposure_path.read_text(encoding="utf-8")
    ).model_copy(update={"evaluated_case_ids_before_failure": ()})
    atomic_write_json_value(exposure_path, exposure.model_dump(mode="json"))
    manifest_path = tmp_path / CASE_REUSE_MANIFEST_NAME
    manifest = DiagnosticCaseReuseManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    ).model_copy(update={"exposure_receipt_sha256": sha256_file(exposure_path)})
    atomic_write_json_value(manifest_path, manifest.model_dump(mode="json"))

    with pytest.raises(ValueError, match="evaluated prefix differs"):
        load_and_verify_case_reuse_bundle(final_path)


def test_reuse_manifest_scopes_raw_collection_diff_to_affected_family(
    tmp_path: Path,
) -> None:
    final_path = _write_bundle(tmp_path)
    path = tmp_path / CASE_REUSE_MANIFEST_NAME
    manifest = DiagnosticCaseReuseManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    payload = manifest.model_dump(mode="json")
    payload["source_changes"][0]["affects_raw_collection"] = True
    payload["source_changes"][0]["affected_families"] = ["elementwise"]

    scoped = DiagnosticCaseReuseManifest.model_validate(payload)

    assert sum(item.disposition == "reuse" for item in scoped.decisions) == 200
    assert load_and_verify_case_reuse_bundle(final_path) == manifest


def test_reuse_manifest_rejects_affected_family_left_reused(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path)
    path = tmp_path / CASE_REUSE_MANIFEST_NAME
    payload = DiagnosticCaseReuseManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    ).model_dump(mode="json")
    payload["source_changes"][0].update(
        {
            "affects_derived_diagnostics": True,
            "affected_families": ["transpose"],
        }
    )

    with pytest.raises(
        ValidationError, match="source-diff-affected families must be replaced"
    ):
        DiagnosticCaseReuseManifest.model_validate(payload)


def test_exposure_receipt_forbids_metric_release() -> None:
    payload = _exposure("c" * 64).model_dump(mode="json")
    payload["released_metric_fields"] = ["median_ape"]

    with pytest.raises(
        ValidationError, match="cannot release acceptance metrics"
    ):
        DiagnosticAcceptanceExposureReceipt.model_validate(payload)
