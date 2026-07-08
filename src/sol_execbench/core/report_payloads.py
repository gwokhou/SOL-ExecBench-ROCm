from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sol_execbench.core.data.path_access import (
    path_bool,
    path_mapping_list,
    path_str_or_none,
)


@dataclass(frozen=True)
class ClaimBoundaryView:
    score_authority: bool = False
    paper_parity: bool = False
    leaderboard_result: bool = False


@dataclass(frozen=True)
class ReportSourceView:
    source_name: str
    schema_version: str | None
    checksum: str | None
    claim_boundary: ClaimBoundaryView


def report_source_view(payload: object, *, source_name: str) -> ReportSourceView:
    return ReportSourceView(
        source_name=source_name,
        schema_version=path_str_or_none(payload, "schema_version"),
        checksum=(
            path_str_or_none(payload, "report_checksum.value")
            or path_str_or_none(payload, "coverage_checksum.value")
            or path_str_or_none(payload, "readiness_checksum.value")
            or path_str_or_none(payload, "inventory_checksum.value")
            or path_str_or_none(payload, "manifest_checksum.value")
        ),
        claim_boundary=ClaimBoundaryView(
            score_authority=path_bool(payload, "claim_boundary.score_authority"),
            paper_parity=path_bool(payload, "claim_boundary.paper_parity"),
            leaderboard_result=path_bool(payload, "claim_boundary.leaderboard_result"),
        ),
    )


def report_record_list(payload: object, path: str) -> list[dict[str, Any]]:
    return path_mapping_list(payload, path)
