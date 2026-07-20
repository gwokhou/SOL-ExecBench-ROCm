from __future__ import annotations

import pytest

from solar.einsum.analyzer import normalize_operation_name


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (
            "torch.nn.functional.scaled_dot_product_attention",
            "scaled_dot_product_attention",
        ),
        ("Model.ConvTranspose2d_3", "convtranspose2d"),
        ("torch.bmm", "bmm"),
        ("aten.matmul", "matmul"),
        ("index_add_4", "index_add"),
        ("torch.nn.functional.smooth_l1_loss", "smooth_l1_loss"),
        ("torch.nn.functional.poisson_nll_loss", "poisson_nll_loss"),
        ("Tensor.__and__", "bitwise_and"),
        ("generated_value_add", "add"),
        ("torch.nn.Hardsigmoid", "hardsigmoid"),
        ("torch.nn.functional.log_softmax", "log_softmax"),
        ("torch.einsum", "einsum"),
        ("torch.cumsum", "cumsum"),
        ("module.amax", "max"),
        ("custom.namespace.operation", "operation"),
    ],
)
def test_operation_aliases_are_normalized_by_ordered_rules(
    operation: str,
    expected: str,
) -> None:
    assert normalize_operation_name(operation) == expected
