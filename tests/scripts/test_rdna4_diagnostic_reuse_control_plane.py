"""Focused contracts for multi-family diagnostic held-out replacement."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from sol_execbench.core.bench.performance_model.case_reuse import (
    DiagnosticHeldOutCorpusFragment,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.validation_corpus import (
    BlobArtifactReference,
    DiagnosticValidationCase,
)
from sol_execbench.core.data.json_utils import load_json_file


def _digest(value: int) -> str:
    return f"{value:064x}"


def _case(family: WorkloadKind, index: int) -> DiagnosticValidationCase:
    identity = list(WorkloadKind).index(family) * 100 + index + 1
    return DiagnosticValidationCase(
        case_id=f"held_out-{family.value}-{index:02d}",
        pair_id=_digest(identity),
        workload_kind=family,
        evidence_manifest=BlobArtifactReference(
            sha256=_digest(identity + 1_000),
            size_bytes=100,
            tree_manifest_sha256=_digest(identity + 2_000),
        ),
        solar_manifest=BlobArtifactReference(
            sha256=_digest(identity + 3_000),
            size_bytes=100,
            tree_manifest_sha256=_digest(identity + 4_000),
        ),
    )


def test_freeze_fragment_canonically_combines_selected_families(
    load_script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reuse = load_script(
        "scripts/internal/rdna4/manage_rdna4_diagnostic_reuse.py"
    )
    specs = [
        SimpleNamespace(family=family, index=index)
        for family in (
            WorkloadKind.ELEMENTWISE,
            WorkloadKind.INDEXED_READ,
            WorkloadKind.INDEXED_UPDATE,
        )
        for index in range(2)
    ]
    monkeypatch.setattr(
        reuse.collector,
        "_require_frozen_design",
        lambda _root: SimpleNamespace(universe_start=460),
    )
    monkeypatch.setattr(
        reuse.collector,
        "_cases",
        lambda _role, _start: specs,
    )
    monkeypatch.setattr(
        reuse.collector,
        "_validation_case",
        lambda _root, spec: _case(spec.family, spec.index),
    )
    (tmp_path / "design.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "fragment.json"

    reuse._freeze_fragment(
        tmp_path,
        (WorkloadKind.INDEXED_READ, WorkloadKind.ELEMENTWISE),
        output,
    )

    fragment = load_json_file(DiagnosticHeldOutCorpusFragment, output)
    assert [case.workload_kind for case in fragment.cases] == [
        WorkloadKind.ELEMENTWISE,
        WorkloadKind.ELEMENTWISE,
        WorkloadKind.INDEXED_READ,
        WorkloadKind.INDEXED_READ,
    ]


def test_freeze_fragment_rejects_duplicate_family(
    load_script,
    tmp_path: Path,
) -> None:
    reuse = load_script(
        "scripts/internal/rdna4/manage_rdna4_diagnostic_reuse.py"
    )

    with pytest.raises(ValueError, match="families must be unique"):
        reuse._freeze_fragment(
            tmp_path,
            (WorkloadKind.ELEMENTWISE, WorkloadKind.ELEMENTWISE),
            tmp_path / "fragment.json",
        )
