from __future__ import annotations

import json

from sol_execbench.core.dataset.execution_closure import (
    build_execution_closure_report,
)
from sol_execbench.core.dataset.long_tail_exclusions import (
    LONG_TAIL_EXCLUSION_REASON,
    LONG_TAIL_EXCLUSION_SCHEMA_VERSION,
    LONG_TAIL_EXCLUSION_STATUS,
    LongTailExclusionConfig,
    exclusion_closure_metadata,
    split_excluded_workloads,
)


def test_long_tail_exclusion_config_matches_workload_and_shard():
    config = LongTailExclusionConfig.model_validate(
        {
            "schema_version": LONG_TAIL_EXCLUSION_SCHEMA_VERSION,
            "exclusions": [
                {
                    "scope": "workload",
                    "problem_id": "L1/example",
                    "workload_uuid": "slow-workload",
                    "reason": "known long-tail workload",
                    "evidence_ref": ".planning/milestones/CDNA3-VALIDATION-HANDOFF.md",
                },
                {
                    "scope": "shard",
                    "problem_id": "L1/example",
                    "shard_index": 2,
                    "reason": "known long-tail shard",
                    "evidence_ref": ".planning/milestones/CDNA3-VALIDATION-HANDOFF.md",
                },
            ],
        }
    )

    selected, excluded = split_excluded_workloads(
        problem_id="L1/example",
        workload_refs=[
            {"uuid": "fast", "row_index": 0},
            {"uuid": "slow-workload", "row_index": 1},
            {"uuid": "shard-match", "row_index": 3},
        ],
        exclusions=config,
        workload_shard_size=2,
    )

    assert selected == [{"uuid": "fast", "row_index": 0}]
    assert [ref["uuid"] for ref, _entry in excluded] == [
        "slow-workload",
        "shard-match",
    ]
    assert config.summary()["claim_boundary"]["excluded_entries_are_passed"] is False


def test_execution_closure_accounts_long_tail_as_non_pass_status():
    entry = LongTailExclusionConfig.model_validate(
        {
            "exclusions": [
                {
                    "scope": "workload",
                    "problem_id": "L1/example",
                    "row_index": 0,
                    "reason": "known long-tail workload",
                    "evidence_ref": ".planning/milestones/CDNA3-VALIDATION-HANDOFF.md",
                }
            ]
        }
    ).exclusions[0]
    metadata = exclusion_closure_metadata(entry)

    report = build_execution_closure_report(
        records=[
            {
                "category": "L1",
                "problem_id": "L1/example",
                "problem_path": "L1/example",
                "workload_uuid": "slow",
                "row_index": 0,
                "closure_status": LONG_TAIL_EXCLUSION_STATUS,
                "filter_reasons": metadata["filter_reasons"],
                "evidence_refs": metadata["evidence_refs"],
                "notes": metadata["notes"],
            }
        ],
        provenance={
            "long_tail_exclusions_path": "long_tail_exclusions.json",
            "long_tail_exclusions_checksum": "sha256",
            "long_tail_exclusions_summary": {"exclusions": 1},
        },
        filters={"long_tail_exclusions": True},
        created_at="2026-06-07T00:00:00Z",
        source_refs={"long_tail_exclusions": "long_tail_exclusions.json"},
    )

    payload = json.loads(report.to_json())
    assert payload["totals"]["records"] == 1
    assert payload["totals"]["attempted"] == 0
    assert payload["totals"]["passed"] == 0
    assert payload["totals"]["failed"] == 0
    assert payload["totals"]["excluded_long_tail"] == 1
    assert payload["records"][0]["closure_status"] == LONG_TAIL_EXCLUSION_STATUS
    assert payload["records"][0]["filter_reasons"] == [LONG_TAIL_EXCLUSION_REASON]
    assert payload["records"][0]["evidence_refs"] == {
        "long_tail_exclusion": ".planning/milestones/CDNA3-VALIDATION-HANDOFF.md"
    }
    assert payload["source_refs"]["long_tail_exclusions"] == (
        "long_tail_exclusions.json"
    )
