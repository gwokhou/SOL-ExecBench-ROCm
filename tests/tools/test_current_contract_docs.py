"""CPU-safe checks for public documentation and evaluator capabilities."""

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

from sol_execbench.core.evaluator_contract import build_evaluator_contract

ROOT = Path(__file__).resolve().parents[2]
_DOC_CHECK = runpy.run_path(str(ROOT / "scripts/check_current_docs.py"))
_unknown_capability_claims = cast(
    Callable[[Path, str], list[str]],
    _DOC_CHECK["_unknown_capability_claims"],
)
_missing_required_references = cast(
    Callable[[Path, str], list[str]],
    _DOC_CHECK["_missing_required_references"],
)


def test_documented_sidecar_capabilities_are_published() -> None:
    capabilities = build_evaluator_contract().capabilities

    assert capabilities["agent_feedback.sidecar"].startswith("optional_")
    assert capabilities["profile_summary.sidecar"].startswith("optional_")


def test_unknown_documented_capability_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "contract.md"
    path.write_text("The `missing.sidecar` capability key is public.\n")

    failures = _unknown_capability_claims(path, path.read_text())

    assert failures == [
        f"{path} claims unpublished evaluator capability 'missing.sidecar'"
    ]


def test_missing_lifecycle_command_reference_is_rejected() -> None:
    path = ROOT / "docs/performance-diagnostics.md"

    failures = _missing_required_references(path, "")

    assert failures
    assert all(
        failure.startswith(
            "docs/performance-diagnostics.md is missing current reference "
        )
        for failure in failures
    )
