from __future__ import annotations

from sol_execbench.core.reports.report_payloads import (
    ClaimBoundaryView,
    ReportSourceView,
    report_record_list,
    report_source_view,
)


def test_report_source_view_normalizes_common_fields() -> None:
    view = report_source_view(
        {
            "schema_version": "demo.v1",
            "report_checksum": {"value": "abc"},
            "claim_boundary": {"score_authority": True, "leaderboard_result": False},
        },
        source_name="demo",
    )

    assert view == ReportSourceView(
        source_name="demo",
        schema_version="demo.v1",
        checksum="abc",
        claim_boundary=ClaimBoundaryView(
            score_authority=True,
            paper_parity=False,
            leaderboard_result=False,
        ),
    )


def test_report_record_list_filters_to_mapping_records() -> None:
    records = report_record_list({"records": [{"id": "a"}, [], {"id": "b"}]}, "records")

    assert records == [{"id": "a"}, {"id": "b"}]
