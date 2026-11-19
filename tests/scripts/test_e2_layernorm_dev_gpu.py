"""CPU checks for the frozen development matrix and permutation comparisons."""

from collections import Counter

import pytest
import torch

from scripts.internal.solar.e2_layernorm_dev_audit import _comparisons
from scripts.internal.solar.e2_layernorm_dev_gpu import COMPLETE_CALLS, roster
from scripts.internal.solar.e2_layernorm_dev_inputs import rotate_inputs
from sol_execbench.core.bench.judge.contracts import JudgeFamily, JudgeTask
from sol_execbench.core.bench.judge.inputs import generate_seeded_inputs


def test_development_roster_retains_all_denominators():
    slots = roster()
    assert Counter(s.session.strategy.value for s in slots) == {
        "p0": 60,
        "p3": 60,
        "p1": 18,
        "p2": 18,
    }
    assert len({s.session.identity for s in slots}) == 156
    assert {s.session.case.instance.value for s in slots} == {"d0", "d1"}
    repeated = slots[:5]
    assert [s.shift for s in repeated] == [0, 0, 0, 1, 17]
    assert len({(s.uuid, s.nonce) for s in repeated}) == 1


def test_rotate_affine_and_values_together():
    src = generate_seeded_inputs(
        JudgeTask.LAYERNORM, JudgeFamily.L3, {"B": 3, "N": 128}, 1701
    )
    shifted = rotate_inputs(src, 17)
    for original, rotated in zip(
        src.values[:3], shifted.values[:3], strict=True
    ):
        assert not torch.equal(original, rotated)
        assert torch.equal(original, rotated.roll(-17, dims=-1))
    assert shifted.values[3] is src.values[3]
    original = src.construction["feature_permutation"]
    assert (
        shifted.construction["feature_permutation"]
        == original[-17:] + original[:-17]
    )


@pytest.mark.parametrize(
    ("condition", "fixed", "varied"),
    [
        ("intact", "passed", "passed"),
        ("same-input-output-drift", "failed", "passed"),
        ("duplicate-permutation", "passed", "failed"),
        ("late-slot-lost", "passed", "indeterminate"),
    ],
    ids=lambda value: value,
)
def test_comparison_distinguishes_repeat_and_permutation(
    condition, fixed, varied
):
    def calls(shift):
        return [
            {
                "input_sha256": (shift, i),
                **{
                    f"unrotated_{name}_sha256": i
                    for name in ("input", "oracle", "candidate", "reference")
                },
            }
            for i in range(COMPLETE_CALLS)
        ]

    call_sets = [calls(n) for n in (0, 0, 0, 1, 17)]
    if condition == "same-input-output-drift":
        call_sets[0][0]["unrotated_candidate_sha256"] = 17
    elif condition == "duplicate-permutation":
        call_sets[3] = calls(0)
    rows = [{"state": "passed", "calls": group} for group in call_sets]
    if condition == "late-slot-lost":
        rows[4] = {"state": "not_reached", "calls": []}
    result = _comparisons(rows)
    assert (result["fixed"], result["permuted"]) == (fixed, varied)
