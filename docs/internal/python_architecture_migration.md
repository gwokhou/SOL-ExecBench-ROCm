# Python architecture migration

First-party dataclasses are slotted and keyword-only. Replace positional
construction such as `EinsumOperand("A", [2, 3])` with explicit construction:

```python
EinsumOperand(name="A", shape=[2, 3])
```

The Einsum operation registry is immutable after construction. Application
composition roots should use `builtin_einsum_registry()`; tests and extensions
should build isolated registries with `EinsumOpRegistryBuilder`, register their
handlers, and call `build()`. The removed `get_global_registry()` and
`register_einsum_op()` APIs have no process-global replacement.

Run `uv run python scripts/check_dataclass_policy.py` after adding or changing
dataclasses. Persisted objects must use an explicit `to_dict()` or their owning
canonical writer rather than `__dict__` reflection.
