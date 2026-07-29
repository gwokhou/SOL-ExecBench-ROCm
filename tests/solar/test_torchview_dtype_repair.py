from solar.ir.extended_einsum.torchview.converter import PyTorchToEinsum


def test_dtype_view_preserves_explicit_destination_dtype() -> None:
    layers = {
        "input": {
            "type": "auxiliary-tensor",
            "output_dtypes": ["torch.float32"],
            "connections": {"inputs": [], "outputs": ["view"]},
        },
        "view": {
            "type": "view",
            "input_dtypes": ["torch.float32"],
            "output_dtypes": ["torch.int32"],
            "module_args": {
                "call_arguments": [
                    {"tensor": 0},
                    {"dtype": "int32"},
                ],
            },
            "connections": {"inputs": ["input"], "outputs": ["view_output"]},
        },
        "view_output": {
            "type": "hidden-tensor",
            "output_dtypes": ["torch.int32"],
            "connections": {"inputs": ["view"], "outputs": []},
        },
    }

    PyTorchToEinsum(strict=True)._repair_torchview_quirks(
        layers,
        ["view"],
        ["input", "view_output"],
    )

    assert layers["view"]["output_dtypes"] == ["torch.int32"]
    assert layers["view_output"]["output_dtypes"] == ["torch.int32"]
