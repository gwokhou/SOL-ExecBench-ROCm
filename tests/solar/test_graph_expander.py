from solar.ir.extended_einsum.torchview.graph_expander import GraphExpander


def test_builtin_handler_uses_canonical_operation_analyzer() -> None:
    expander = GraphExpander(create_cache_dir=False)
    handler = expander.registry.get_handler("matmul")

    assert handler is not None
    generate = handler.generate_einsum_func
    assert generate is not None
    assert generate([2, 3], [3, 4]).equation == "MK,KN->MN"
    assert not hasattr(expander, "expand_complex_operations_in_graph")
