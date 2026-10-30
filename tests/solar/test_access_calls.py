"""Static call decoding preserves edge cases without unbounded allocation."""

import pytest

from solar.ir.access import MAX_ACCESS_OUTPUTS, AccessReason, base_access
from solar.ir.access_calls import derive_accesses
from solar.ir.access_views import UnsupportedAccessError


def test_empty_chunk_preserves_all_requested_output_slots():
    view = base_access("input", (0, 4), "fp32")
    values = derive_accesses(
        view,
        {
            "target": "chunk",
            "operands": [{"tensor": 0}],
            "attributes": {"chunks": {"value": 4}, "dim": {"value": 0}},
        },
        [(0, 4)] * 4,
    )
    assert len(values) == 4
    assert all(value.elements == 0 for value in values)
    assert [value.output_slot for value in values] == [0, 1, 2, 3]


@pytest.mark.parametrize("target", ["split", "chunk", "tensor_split"])
def test_partition_budget_is_checked_before_materializing_outputs(target):
    # split(size=1) exceeds the output budget; the others request too many parts.
    size = MAX_ACCESS_OUTPUTS + 1
    parts = 1 if target == "split" else size
    with pytest.raises(UnsupportedAccessError) as caught:
        derive_accesses(
            base_access("input", (size,), "fp32"),
            {
                "target": target,
                "arguments": [{"tensor": 0}, {"value": parts}],
                "kwargs": {},
            },
            [],
        )
    assert caught.value.reason is AccessReason.REGION_BUDGET


def test_partition_output_arity_and_shape_must_match_parameters():
    with pytest.raises(ValueError, match="disagrees with IR output"):
        derive_accesses(
            base_access("input", (8,), "fp32"),
            {
                "target": "split",
                "operands": [{"tensor": 0}, {"value": 2}],
                "attributes": {},
            },
            [(3,), (5,)],
        )


def test_large_requested_chunk_count_with_two_actual_outputs_is_cheap():
    values = derive_accesses(
        base_access("input", (2,), "fp32"),
        {
            "target": "chunk",
            "arguments": [{"tensor": 0}, {"value": 10**12}],
            "kwargs": {},
        },
        [(1,), (1,)],
    )
    assert [value.offset for value in values] == [0, 1]


@pytest.mark.parametrize(
    "shape,target,args,output,strides",
    [
        ((2, 1, 3), "squeeze", [], (2, 3), (3, 1)),
        ((2, 3), "unsqueeze", [1], (2, 1, 3), (3, 1, 1)),
        ((2, 3, 4), "flatten", [1, 2], (2, 12), (12, 1)),
        ((2, 3, 4), "permute", [2, 0, 1], (4, 2, 3), (1, 12, 4)),
        ((), "transpose", [], (), ()),
        ((4,), "transpose", [], (4,), (1,)),
    ],
    ids=[
        "remove-singleton",
        "insert-singleton",
        "flatten-contiguous-axes",
        "permute-varargs",
        "scalar-t",
        "vector-t",
    ],
)
def test_static_view_parameter_forms_preserve_logical_offsets(
    shape, target, args, output, strides
):
    values = derive_accesses(
        base_access("input", shape, "fp32"),
        {
            "target": target,
            "arguments": [{"tensor": 0}, *({"value": value} for value in args)],
            "kwargs": {},
        },
        [output],
    )
    assert values[0].strides == strides
    assert values[0].offset == 0
