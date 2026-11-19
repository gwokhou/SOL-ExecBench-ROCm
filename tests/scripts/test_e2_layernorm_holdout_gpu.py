"""CPU checks for the frozen LayerNorm h0 matrix, custody, and audit."""

from collections import Counter
from pathlib import Path

import pytest

from scripts.internal.solar.e2_contracts import base_seed
from scripts.internal.solar.e2_layernorm_holdout_audit import (
    _comparisons,
    verify_custody,
)
from scripts.internal.solar.e2_layernorm_holdout_gpu import (
    CLOSED_KEY_FILE,
    COMMITMENT_KIND,
    COMPLETE_CALLS,
    PROTOCOL,
    HoldoutLane,
    roster,
    roster_digest,
)
from scripts.internal.solar.e2_layernorm_holdout_inputs import (
    LayerNormHoldoutService,
)
from scripts.internal.solar.e2_successor_exposure import COMMITMENT_FILE
from sol_execbench.core.bench.judge.contracts import JudgeInstance, JudgeTask
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    canonical_json_bytes,
)
from sol_execbench.core.integrity import sha256_bytes, sha256_file


def test_holdout_roster_retains_all_denominators():
    slots = roster()
    assert len(slots) == 186
    assert Counter(s.session.strategy.value for s in slots) == {
        "p0": 84,
        "p3": 84,
        "p1": 9,
        "p2": 9,
    }
    assert len({s.session.identity for s in slots}) == 186
    assert {s.session.case.instance.value for s in slots} == {"h0"}

    group = slots[:7]
    assert [s.shift for s in group] == [0, 0, 0, 1, 17, 0, 0]
    assert [s.lane for s in group] == [
        *[HoldoutLane.CONTROL] * 5,
        *[HoldoutLane.CONTROL_DRAW] * 2,
    ]
    assert len({(s.uuid, s.nonce) for s in group[:5]}) == 1
    assert len({(s.uuid, s.nonce) for s in group[4:]}) == 3

    negatives = [s for s in slots if s.lane is HoldoutLane.NEGATIVE]
    assert len(negatives) == 18
    assert len({(s.uuid, s.nonce) for s in negatives[:3]}) == 3


@pytest.mark.parametrize(
    ("condition", "fixed", "permuted", "draw"),
    [
        ("intact", "passed", "passed", "passed"),
        ("same-byte-drift", "failed", "passed", "passed"),
        ("duplicate-permutation", "passed", "failed", "passed"),
        ("duplicate-independent-draw", "passed", "passed", "failed"),
        ("unreached-draw", "passed", "passed", "indeterminate"),
    ],
    ids=lambda value: value,
)
def test_holdout_comparisons_distinguish_three_consistency_checks(
    condition, fixed, permuted, draw
):
    def calls(tag, *, distinct_prefix=0):
        return [
            {
                "input_sha256": (distinct_prefix, tag, i),
                **{
                    f"unrotated_{name}_sha256": (distinct_prefix, i)
                    for name in ("input", "oracle", "candidate", "reference")
                },
            }
            for i in range(COMPLETE_CALLS)
        ]

    call_sets = [
        *[calls("anchor", distinct_prefix=0) for _ in range(3)],
        calls("shift-1", distinct_prefix=0),
        calls("shift-17", distinct_prefix=0),
        calls("draw-5", distinct_prefix=5),
        calls("draw-6", distinct_prefix=6),
    ]
    if condition == "same-byte-drift":
        call_sets[0][0]["unrotated_candidate_sha256"] = (0, 99)
    elif condition == "duplicate-permutation":
        call_sets[3] = calls("anchor", distinct_prefix=0)
    elif condition == "duplicate-independent-draw":
        call_sets[5] = calls("anchor", distinct_prefix=0)
    rows = [{"state": "passed", "calls": group} for group in call_sets]
    if condition == "unreached-draw":
        rows[6] = {"state": "not_reached", "calls": []}
    result = _comparisons(rows)
    assert (
        result["fixed"],
        result["permuted"],
        result["independent_draw"],
    ) == (fixed, permuted, draw)


def test_holdout_service_enforces_h0_and_commitment_custody(tmp_path: Path):
    dev_stage = tmp_path / "dev-stage"
    dev_stage.mkdir()
    atomic_write_json_value(
        dev_stage / "judge-private.json",
        {
            "task": JudgeTask.LAYERNORM.value,
            "instance": JudgeInstance.DEVELOPMENT_0.value,
            "feature_shift": 0,
        },
    )
    with pytest.raises(ValueError, match="accepts h0 only"):
        LayerNormHoldoutService(dev_stage)

    bad_stage = tmp_path / "outside"
    bad_stage.mkdir()
    atomic_write_json_value(
        bad_stage / "judge-private.json",
        {
            "task": JudgeTask.LAYERNORM.value,
            "instance": JudgeInstance.HOLDOUT.value,
            "feature_shift": 0,
            "holdout_commitment_sha256": "a" * 64,
        },
    )
    with pytest.raises(ValueError, match="outside the committed online"):
        LayerNormHoldoutService(bad_stage)

    work = tmp_path / "holdout-confirmation"
    work.mkdir()
    roster_rows = [slot.metadata() for slot in roster()]
    (work / "roster.json").write_bytes(canonical_json_bytes(roster_rows))
    roster_sha = roster_digest()
    key = bytes(range(32))
    freeze_sha = "f" * 64
    commitment_path = work / COMMITMENT_FILE
    atomic_write_json_value(
        commitment_path,
        {
            "research_format": COMMITMENT_KIND,
            "protocol": PROTOCOL,
            "key_sha256": sha256_bytes(key),
            "freeze_sha256": freeze_sha,
            "roster_sha256": roster_sha,
            "schedule_sha256": roster_sha,
            "instances_materialized_at_commitment": 0,
            "performance_authority": False,
        },
    )
    commitment_sha = sha256_file(commitment_path)
    atomic_write_json_value(
        work / "online-summary.json",
        {
            "freeze_sha256": freeze_sha,
            "commitment_sha256": commitment_sha,
            "roster_sha256": roster_sha,
        },
    )
    (work / CLOSED_KEY_FILE).write_bytes(key)
    spec = {"roster_sha256": roster_sha}
    loaded_key, loaded_commitment = verify_custody(work, spec, freeze_sha)
    assert loaded_key == key
    assert loaded_commitment == commitment_sha
    assert base_seed(roster()[0].session.case, loaded_key) >= 0

    (work / CLOSED_KEY_FILE).write_bytes(bytes(32))
    with pytest.raises(ValueError, match="post-online key mismatch"):
        verify_custody(work, spec, freeze_sha)
